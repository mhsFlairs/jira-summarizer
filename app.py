import os
import logging
import streamlit as st
from dotenv import load_dotenv
from jira import JIRA
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import pandas as pd
from datetime import datetime
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# === Configuration ===
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Constants
ITEMS_PER_PAGE = 10
MAX_HISTORY_ITEMS = 50
MAX_QUERY_LENGTH = 500


def initialize_session_state():
    """
    Initializes or resets session state variables.
    """
    default_states = {
        "query_history": [],
        "JQL_QUERY": "",
        "max_results": 100,
        "timeout": 30,
        "temperature": 0.0,
        "page_number": 1,
        "chat_enabled": False,
        "current_tickets": None,  # Add this line
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def generate_stakeholder_report(tickets, conversation_history=None):
    """
    Generates a comprehensive stakeholder report based on the provided tickets.

    Args:
        tickets: JIRA ticket objects to analyze
        conversation_history: Optional previous conversation for context

    Returns:
        str: A formatted stakeholder report
    """
    try:
        if not tickets:
            return "No tickets available to generate a report."

        # Extract relevant ticket data
        ticket_data = [
            extract_ticket_data(ticket, include_details=True) for ticket in tickets
        ]
        ticket_data = [data for data in ticket_data if data is not None]

        # Format tickets for the prompt
        formatted_tickets = []
        for data in ticket_data:
            ticket_info = [
                f"Key: {data['Key']}",
                f"Summary: {data['Summary']}",
                f"Status: {data['Status']}",
                f"Priority: {data['Priority']}",
                f"Assignee: {data['Assignee']}",
                f"Updated: {data['Updated']}",
            ]

            if "Description" in data and data["Description"]:
                ticket_info.append(f"Description: {data['Description']}")

            if "Comments" in data and data["Comments"]:
                comments_text = "\n".join(
                    [f"- {comment}" for comment in data["Comments"][:3]]
                )
                ticket_info.append(f"Recent Comments: {comments_text}")

            formatted_tickets.append("\n".join(ticket_info))

        tickets_text = "\n\n".join(formatted_tickets)

        # Include conversation context if available
        context = ""
        if conversation_history:
            # Format the last few conversation turns for context
            context_messages = (
                conversation_history[-3:]
                if len(conversation_history) > 3
                else conversation_history
            )
            context = "\n\nConversation context:\n" + "\n".join(
                [
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in context_messages
                ]
            )

        # Return the formatted tickets for the agent to process
        return f"""Here are the JIRA tickets to analyze for your stakeholder report:
        
{tickets_text}
{context}

Based on the above information, please create a comprehensive stakeholder report without asking for additional information.
"""
    except Exception as e:
        logger.error(f"Error generating stakeholder report data: {e}")
        return f"Error generating stakeholder report: {str(e)}"


def get_jira_connection():
    """
    Creates and returns a JIRA connection.
    Returns: JIRA connection object
    """
    try:
        return JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
    except Exception as e:
        logger.error(f"Failed to establish JIRA connection: {e}")
        raise


def validate_jql(jql: str) -> bool:
    """
    Validates JQL syntax.
    """
    if not jql or not jql.strip():
        st.warning("Please enter a valid JQL query")
        logger.warning("Empty JQL query submitted")
        return False

    try:
        logger.info(f"Validating JQL: {jql}")
        jira = get_jira_connection()
        jira.search_issues(jql, maxResults=1)
        return True
    except Exception as e:
        logger.error(f"Invalid JQL: {e}")
        st.error(f"Invalid JQL: {str(e)}")
        return False


def load_jira_tickets(jql_query: str, max_results: int, expanded: bool = False):
    """
    Common function to load JIRA tickets.
    """
    try:
        logger.info(f"Fetching tickets with JQL: {jql_query}")
        jira = get_jira_connection()

        expand_params = ["renderedFields"] if expanded else None
        issues = jira.search_issues(
            jql_query, maxResults=max_results, expand=expand_params
        )

        logger.info(f"Successfully fetched {len(issues)} tickets")
        return issues
    except Exception as e:
        logger.error(f"Error loading tickets: {e}")
        raise


def extract_ticket_data(issue, include_details: bool = False):
    """
    Extracts relevant data from a JIRA issue.
    """
    try:
        data = {
            "Key": issue.key,
            "Summary": issue.fields.summary,
            "Status": issue.fields.status.name,
            "Priority": issue.fields.priority.name,
            "Assignee": (
                str(issue.fields.assignee) if issue.fields.assignee else "Unassigned"
            ),
            "Updated": issue.fields.updated[:10],
        }

        if include_details:
            data.update(
                {
                    "Description": issue.fields.description,
                    "Comments": (
                        [c.body for c in issue.fields.comment.comments]
                        if hasattr(issue.fields, "comment")
                        else []
                    ),
                    "Resolution": (
                        str(issue.fields.resolution)
                        if issue.fields.resolution
                        else None
                    ),
                }
            )

        return data
    except Exception as e:
        logger.error(f"Error extracting ticket data for {issue.key}: {e}")
        return None


def initialize_chat_agent():
    """
    Initializes the chat agent using Azure OpenAI.
    """
    try:
        chat_model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version="2023-05-15",
            temperature=0.7,
        )
        return chat_model
    except Exception as e:
        logger.error(f"Failed to initialize chat agent: {e}")
        return None


# 1. Fix the system_message in create_jira_agent function (convert string to SystemMessage)


# 2. Update the create_jira_agent function to pass conversation history to the tool
def create_jira_agent(chat_model, tickets):
    """
    Creates an agent with tools for analyzing JIRA tickets.
    """
    try:
        # Define tools that pass conversation history
        tools = [
            Tool(
                name="generate_stakeholder_report",
                func=lambda _: generate_stakeholder_report(
                    tickets, st.session_state.get("agent_messages", [])
                ),
                description="Generates a comprehensive stakeholder report based on the current JIRA tickets. Use this when asked to create, generate, or prepare a report for stakeholders or management.",
            )
        ]

        # Define agent prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful JIRA analysis assistant that can discuss Jira tickets and provide insights about them.
When asked to generate a stakeholder report, use the generate_stakeholder_report tool to get ticket data, then analyze it and create a professional report with the following sections:

1. Executive Summary
- High-level overview of project progress
- Key achievements in this period
- Critical milestones reached

2. Project Status Overview
- Overall project health (On Track/At Risk/Delayed)
- Current phase of the project
- Percentage of completion against planned timeline

3. Key Deliverables Status
- Completed deliverables with brief descriptions
- In-progress items with expected completion dates
- Upcoming major deliverables
- Any changes to agreed scope

4. Risk and Issues
- Current blocking issues
- Potential risks identified
- Mitigation strategies in place

5. Timeline and Milestones
- Major milestones achieved
- Next important dates
- Any schedule variations

6. Resource Utilization
- Team capacity and allocation
- Any resource constraints or needs

7. Next Steps
- Priority items for next period
- Required decisions or support needed
- Upcoming key activities

Use clear, professional language and format the report in a way that's easily digestible for senior stakeholders.
Include relevant metrics and KPIs where available. Highlight any items requiring immediate attention or executive decision-making.

IMPORTANT: The user has already provided all necessary information about the tickets. Do not ask for additional information or clarification - use what is available to create the best possible report.
""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create agent
        agent = create_openai_tools_agent(chat_model, tools, prompt)

        # Set up memory with existing conversation if available
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Pre-populate memory with existing conversation
        if "agent_messages" in st.session_state and st.session_state.agent_messages:
            for message in st.session_state.agent_messages:
                if message["role"] == "user":
                    memory.chat_memory.add_user_message(message["content"])
                else:
                    memory.chat_memory.add_ai_message(message["content"])

        # Create agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )

        return agent_executor

    except Exception as e:
        logger.error(f"Error creating JIRA agent: {e}")
        return None


# 2. Fix the chat_about_tickets function to remove duplicate buttons
# 2. Fix the chat_about_tickets function to use invoke instead of run
def chat_about_tickets(tickets, chat_model):
    """
    Handles chat interactions about the displayed tickets using an agent with tools.
    """
    # Create the agent if not already in session state
    if "agent" not in st.session_state:
        agent = create_jira_agent(chat_model, tickets)
        if agent is None:
            st.error("Failed to create agent. Please try again.")
            return
        st.session_state.agent = agent

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Create a container for the chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.agent_messages:
            with st.chat_message("user" if message["role"] == "user" else "assistant"):
                st.write(message["content"])

        # Get user input at the bottom
        prompt = st.chat_input(
            "Ask about tickets or request a stakeholder report...",
            key="agent_chat_input",
        )

        if prompt:
            # Add user message to chat history
            st.session_state.agent_messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Process with agent
            with st.spinner("Thinking..."):
                try:
                    # Check if agent exists and is valid
                    if st.session_state.agent is None:
                        raise ValueError("Agent is not initialized")

                    # Use invoke instead of run and extract the text content
                    response = st.session_state.agent.invoke({"input": prompt})
                    response_text = response.get(
                        "output", "I'm sorry, I couldn't generate a response."
                    )

                    # Add AI response to chat history
                    st.session_state.agent_messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Display AI response
                    with st.chat_message("assistant"):
                        st.write(response_text)

                        # If this is a stakeholder report, offer download option
                        # But don't reset the conversation - allow continued refinement
                        if (
                            "stakeholder report" in prompt.lower()
                            and len(response_text) > 500
                        ):
                            # Generate a unique key for each download button based on timestamp
                            download_key = f"download_report_button_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                            st.download_button(
                                label="Download Report",
                                data=response_text,
                                file_name=f"stakeholder_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                mime="text/markdown",
                                key=download_key,
                            )

                            # Add a helpful hint for users to continue the conversation
                            st.info(
                                "You can continue the conversation to refine this report. Try asking for specific changes or additions."
                            )

                except Exception as e:
                    logger.error(f"Error in agent response: {e}")
                    st.error(f"Failed to get response from the agent: {str(e)}")

                    # Try to reinitialize the agent
                    st.warning("Attempting to reinitialize the agent...")
                    if "agent" in st.session_state:
                        del st.session_state.agent
                    new_agent = create_jira_agent(chat_model, tickets)
                    if new_agent:
                        st.session_state.agent = new_agent
                        st.success(
                            "Agent reinitialized. Please try your question again."
                        )
                    else:
                        st.error(
                            "Could not reinitialize the agent. Please restart the chat."
                        )


# 4. Fix the display_tickets_table function with proper key for the button
def display_tickets_table(tickets):
    """
    Displays tickets in a table format and provides chat option.
    """
    try:
        if not tickets:
            st.warning("No tickets found for the current query.")
            return

        data = [extract_ticket_data(issue) for issue in tickets]
        data = [d for d in data if d is not None]

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.info(f"Total tickets found: {len(data)}")
        logger.info(f"Displayed {len(data)} tickets in table format")

        # Store tickets in session state
        st.session_state.current_tickets = tickets

        # Add chat option
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button(
                "Start Chat" if not st.session_state.chat_enabled else "End Chat",
                key="toggle_chat_button",  # Add unique key to fix the button ID error
            ):
                st.session_state.chat_enabled = not st.session_state.chat_enabled
                # Reset agent when toggling chat
                if "agent" in st.session_state:
                    del st.session_state.agent
                if "agent_messages" in st.session_state:
                    st.session_state.agent_messages = []
                st.rerun()

        if st.session_state.chat_enabled and st.session_state.current_tickets:
            chat_model = initialize_chat_agent()
            if chat_model:
                st.write(
                    "You can now chat about the tickets or ask for a stakeholder report:"
                )
                st.info(
                    "Try asking: 'Please prepare a stakeholder report' or 'Generate a report for management'"
                )
                chat_about_tickets(st.session_state.current_tickets, chat_model)
            else:
                st.error("Failed to initialize chat agent")

    except Exception as e:
        logger.error(f"Error displaying tickets table: {e}")
        st.error(f"Error displaying tickets table: {str(e)}")


def manage_query_history():
    """
    Manages query history size.
    """
    if len(st.session_state.query_history) > MAX_HISTORY_ITEMS:
        logger.info(f"Trimming query history to {MAX_HISTORY_ITEMS} items")
        st.session_state.query_history = st.session_state.query_history[
            -MAX_HISTORY_ITEMS:
        ]


def save_to_history(jql, response):
    """
    Saves JQL query and response to history.
    """
    st.session_state.query_history.append(
        {
            "jql": jql,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    manage_query_history()


def main():
    try:
        st.title("Jira Project Analyst")

        # Initialize session state
        initialize_session_state()

        # Sidebar settings
        with st.sidebar:
            st.header("Settings")

            if st.button("Clear History"):
                st.session_state.chat_enabled = False
                st.session_state.current_tickets = None
                initialize_session_state()
                st.success("History cleared")
                st.rerun()

            st.session_state.timeout = st.slider("Query Timeout (seconds)", 10, 60, 30)
            st.session_state.max_results = st.number_input("Max Results", 10, 500, 100)

        # JQL input
        jql_query = st.text_area("Enter JQL Query:", height=100)

        if st.button("Execute JQL"):
            st.session_state.chat_enabled = (
                False  # Reset chat when new query is executed
            )
            if validate_jql(jql_query):
                st.session_state.JQL_QUERY = jql_query
                tickets = load_jira_tickets(jql_query, st.session_state.max_results)
                display_tickets_table(tickets)
                save_to_history(jql_query, "Query executed successfully")

        # If there are current tickets, display them and the chat interface if enabled
        elif st.session_state.current_tickets:
            display_tickets_table(st.session_state.current_tickets)

        # Query History
        with st.expander("Query History"):
            for item in reversed(st.session_state.query_history):
                st.markdown(f"**JQL ({item['timestamp']}):** {item['jql']}")
                st.markdown(f"**Response:** {item['response']}")
                st.markdown("---")

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    main()
