import os
import logging
import streamlit as st
from dotenv import load_dotenv
from jira import JIRA
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
load_dotenv(override=True)

# === Configuration ===
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Constants
MAX_HISTORY_ITEMS = 50


def initialize_session_state():
    """
    Initializes or resets session state variables.
    """
    default_states = {
        "query_history": [],
        "JQL_QUERY": "",
        "max_results": 1000,
        "timeout": 30,
        "temperature": 0.0,
        "page_number": 1,
        "chat_enabled": False,
        "current_tickets": None,  # Add this line
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def generate_stakeholder_report(tickets_text, conversation_history=None):
    """
    Generates a comprehensive stakeholder report based on the provided tickets.

    Args:
        tickets: JIRA ticket objects to analyze
        conversation_history: Optional previous conversation for context

    Returns:
        str: A formatted stakeholder report
    """
    try:
        if not tickets_text:
            return "No tickets available to generate a report."

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

        # Return instructions for the agent to create the report
        return f"""Based on the JIRA tickets that have already been provided to you, please create a comprehensive stakeholder report with the following sections:

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

                Conversation context:
                {context}
                Jira Tickets:
                {tickets_text}

                IMPORTANT: You already have access to all the ticket information. Do not ask for additional information - use what is available to create the best possible report.
"""
    except Exception as e:
        logger.error(f"Error generating stakeholder report data: {e}")
        return f"Error generating stakeholder report: {str(e)}"


def generate_summary(tickets_text, conversation_history=None):
    """
    Generates a summary of the provided tickets.

    Args:
        tickets: JIRA ticket objects to summarize
        conversation_history: Optional previous conversation for context

    Returns:
        str: A formatted summary of the tickets
    """
    try:
        if not tickets_text:
            return "No tickets available to summarize."

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

        # Return instructions for the agent to create the summary
        return f"""
    
        Prompt:

        You are a technical product analyst. Based on the following Jira tickets, generate a concise product summary. Focus on the product's purpose, key features, recent changes, and target users. Eliminate redundant or low-impact information. Structure the summary in clear paragraphs or bullet points.

        Conversation Context:
        {context}

        Jira Tickets:

        {tickets_text}
        
        IMPORTANT: You already have access to all the ticket information. Do not ask for additional information - use what is available to create the best possible summary.
        """
    except Exception as e:
        logger.error(f"Error generating summary data: {e}")
        return f"Error generating summary: {str(e)}"


def generate_product_documentation(tickets_text, conversation_history=None):
    """
    Generates product documentation based on the provided tickets.

    Args:
        tickets: JIRA ticket objects to analyze
        conversation_history: Optional previous conversation for context

    Returns:
        str: A formatted product documentation
    """
    try:
        if not tickets_text:
            return "No tickets available to generate documentation."

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

        # Return instructions for the agent to create the report
        return f"""
                Generate comprehensive technical product documentation for a software platform. The report must be detailed, formal, and formatted using the following structure. All relevant feature IDs (e.g., JIRA ticket IDs), external references (Confluence pages, design files, repositories), and supporting links (e.g., API docs, videos, screenshots) must be included in appropriate sections.

                **Document Structure:**

                1. **Executive Summary**
                    - High-level overview of the platform's purpose, scope, and key benefits.
                    - Mention key stakeholders (e.g., product owners, admins, IT, end users).
                2. **Product Vision & Objectives**
                    - Clear vision statement.
                    - List strategic objectives (e.g., support accuracy, access control, automation).
                3. **Core Features & Capabilities**
                    - Organize features by functional area (e.g., Chat, Admin, Knowledge Base).
                    - For each feature:
                        - Brief description of purpose and behavior.
                        - Associated ticket IDs (e.g., PROJ-123, PROJ-456).
                        - External links (e.g., design mockups, technical specs, APIs).
                4. **User Roles & Permissions**
                    - Define all user roles (Admin, Manager, Contributor, End User, etc.).
                    - Detail permissions by role.
                    - Include access scope, role mapping logic, and data filtration mechanisms.
                5. **Business Processes & Workflows**
                    - Describe step-by-step workflows (e.g., onboarding, knowledge ingestion, chat flow).
                    - Indicate where automation is implemented.
                    - Link to relevant process diagrams or Confluence pages.
                6. **Technical Architecture Overview**
                    - Outline major system components (Frontend, Backend, Database, Integrations).
                    - Describe authentication and data flow.
                    - Reference diagrams or architecture documents (e.g., system blueprint links).
                7. **API Documentation**
                    - List key endpoints, grouped by functional domain.
                    - Include methods (GET, POST, etc.), required tokens, and access logic.
                    - Link to Swagger/OpenAPI docs and authentication references.
                8. **UI/UX Guidelines**
                    - Summarize design principles used.
                    - Mention interaction patterns, accessibility standards, and error handling.
                    - Reference Figma links or design system documentation.
                9. **Release Notes & Change History**
                    - List major released features with ticket IDs.
                    - Identify canceled items and UAT-specific features.
                    - Include version info and dates if available.
                10. **Troubleshooting & Known Issues**
                    - List ongoing blockers and known issues with associated ticket IDs.
                    - Link to relevant bug reports, logs, or tracking boards (e.g., JIRA filters).
                11. **References & Supporting Links**
                    - Consolidate all external documentation:
                        - Confluence articles
                        - Figma files
                        - API portals
                        - Repositories (e.g., Bitbucket, GitHub)
                        - Videos/images tied to feature delivery or UAT

                **Requirements:**

                - The tone should be formal and objective.
                - Use bullet points and subheadings for clarity.
                - Include exact ticket references and documentation links wherever possible.
                - Structure the content for internal distribution to technical stakeholders, project managers, and support leads.

                Conversation context:
                {context}
                Jira Tickets:
                {tickets_text}
                """
    except Exception as e:
        logger.error(f"Error generating product documentation data: {e}")
        return f"Error generating product documentation: {str(e)}"


def get_jira_connection():
    """
    Creates and returns a JIRA connection.
    Returns: JIRA connection object
    """
    try:
        logger.info("Connecting to JIRA...")
        logger.info(f"Using JIRA server: {JIRA_SERVER}")
        logger.info(f"Using JIRA user: {JIRA_USER}")
        logger.info(f"Using JIRA API token: {JIRA_API_TOKEN}")
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
        jira = get_jira_connection()
        logger.info(f"Validating JQL: {jql}")
        jira.search_issues(jql, maxResults=1)
        return True
    except Exception as e:
        logger.error(f"Invalid JQL: {e}")
        st.error(f"Invalid JQL: {str(e)}")
        return False


def load_jira_tickets(
    jql_query: str,
    max_results: int = None,
    expanded: bool = False,
    load_all: bool = False,
):
    """
    Common function to load JIRA tickets with pagination support.

    Args:
        jql_query: JQL query string
        max_results: Maximum number of results to return (None for unlimited when load_all=True)
        expanded: Whether to expand fields
        load_all: If True, loads all tickets using pagination
    """
    try:
        logger.info(f"Fetching tickets with JQL: {jql_query}")
        jira = get_jira_connection()

        if not load_all:
            # Original behavior - load up to max_results
            expand_params = ["renderedFields"] if expanded else None
            issues = jira.search_issues(
                jql_query, maxResults=max_results, expand=expand_params
            )
            logger.info(f"Successfully fetched {len(issues)} tickets")
            return issues

        # Pagination logic for loading all tickets
        all_issues = []
        start_at = 0
        page_size = 100  # Jira's recommended page size
        total_issues = None

        # Show progress bar in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            try:
                expand_params = ["renderedFields"] if expanded else None

                # Fetch a page of results
                page_issues = jira.search_issues(
                    jql_query,
                    startAt=start_at,
                    maxResults=page_size,
                    expand=expand_params,
                )

                # If this is the first request, get the total count
                if total_issues is None:
                    # Get total count by searching with maxResults=0
                    count_result = jira.search_issues(jql_query, maxResults=0)
                    total_issues = count_result.total
                    logger.info(f"Total tickets to fetch: {total_issues}")

                    if total_issues == 0:
                        logger.info("No tickets found")
                        break

                    # Apply max_results limit if specified
                    if max_results and max_results < total_issues:
                        total_issues = max_results
                        logger.info(
                            f"Limiting to {max_results} tickets as per max_results setting"
                        )

                # Add the issues from this page
                current_page_count = len(page_issues)
                all_issues.extend(page_issues)

                # Update progress
                progress = min(len(all_issues) / total_issues, 1.0)
                progress_bar.progress(progress)
                status_text.text(
                    f"Loaded {len(all_issues)} of {total_issues} tickets..."
                )

                logger.info(
                    f"Fetched page starting at {start_at}, got {current_page_count} tickets. Total so far: {len(all_issues)}"
                )

                # Check if we've reached the end or our limit
                if (
                    current_page_count < page_size
                    or len(all_issues) >= total_issues
                    or (max_results and len(all_issues) >= max_results)
                ):
                    break

                start_at += page_size

            except Exception as page_error:
                logger.error(
                    f"Error fetching page starting at {start_at}: {page_error}"
                )
                # If we have some results, return them; otherwise re-raise
                if all_issues:
                    logger.warning(
                        f"Partial results returned: {len(all_issues)} tickets"
                    )
                    break
                else:
                    raise page_error

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        # Apply max_results limit if specified
        if max_results and len(all_issues) > max_results:
            all_issues = all_issues[:max_results]
            logger.info(f"Trimmed results to {max_results} tickets")

        logger.info(f"Successfully fetched {len(all_issues)} tickets using pagination")
        return all_issues

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


def escape_braces(s):
    return s.replace("{", "{{").replace("}", "}}")


def create_tickets_text(tickets):
    # Extract and format ticket data at agent creation time
    try:
        logger.info("Creating tickets text for system prompt...")
        ticket_data = [
            extract_ticket_data(ticket, include_details=True) for ticket in tickets
        ]
        ticket_data = [data for data in ticket_data if data is not None]

        # Format tickets for the system prompt
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

        # stringify the tickets text for the system prompt
        tickets_text = escape_braces(tickets_text)
        return tickets_text
    except Exception as e:
        logger.error(f"Error creating tickets text: {e}")
        raise e


# 2. Update the create_jira_agent function to pass conversation history to the tool
def create_jira_agent(chat_model, tickets):
    """
    Creates an agent with tools for analyzing JIRA tickets.
    """
    try:

        tickets_text = create_tickets_text(tickets)

        logger.info("Creating JIRA agent with tools...")
        # Define tools
        tools = [
            Tool(
                name="generate_stakeholder_report",
                func=lambda _: generate_stakeholder_report(
                    tickets_text, st.session_state.get("agent_messages", [])
                ),
                description="""Only use this for stakeholders or client not anyone who is internal. Generates a comprehensive stakeholder report based on the current JIRA tickets. Use this when asked to create, generate, or prepare a report for stakeholders or management.""",
            ),
            Tool(
                name="summarize_tickets",
                func=lambda _: generate_summary(
                    tickets_text, st.session_state.get("agent_messages", [])
                ),
                description="""Summarizes the current JIRA tickets. Use this when asked to provide a summary of the tickets or project status""",
            ),
            Tool(
                name="generate_product_documentation",
                func=lambda _: generate_product_documentation(
                    tickets_text, st.session_state.get("agent_messages", [])
                ),
                description="""Generates product documentation based on the current JIRA tickets. Use this when asked to create or generate product documentation.""",
            ),
        ]

        logger.info(f"Defined {len(tools)} tools for the agent")

        # Define agent prompt with ticket data included in system message
        system_message = f"""You are a helpful JIRA analysis assistant that can discuss Jira tickets and provide insights about them.
        You will be given jira tickets, You can answer questions about these tickets, provide analysis, summaries, and generate stakeholder reports. When users ask about tickets, refer to the data above. When asked to generate a stakeholder report, use the generate_stakeholder_report tool."""

        tickets_context_message = f"""You have access to the following JIRA tickets:
        {tickets_text}
        Use this information to answer questions, provide analysis, and generate reports as needed.
        """

        logger.info("Creating chat prompt template...")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("system", tickets_context_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        logger.info("Creating agent with tools...")

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

        logger.info("Agent and memory created successfully")

        logger.info("Creating agent executor...")
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
                # Use markdown to make text selectable/copyable
                st.markdown(message["content"])

                # If this was an assistant message with a stakeholder report, add download button
                if (
                    message["role"] == "assistant"
                    and any(
                        term in st.session_state.agent_messages[-2]["content"].lower()
                        for term in [
                            "stakeholder report",
                            "generate report",
                            "create report",
                        ]
                    )
                    if len(st.session_state.agent_messages) >= 2
                    else False and len(message["content"]) > 500
                ):
                    # Generate a unique key for each download button based on message index
                    download_key = f"download_report_button_{st.session_state.agent_messages.index(message)}"

                    # Add copy button for the content
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.download_button(
                            label="Download Report",
                            data=message["content"],
                            file_name=f"stakeholder_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            key=download_key,
                        )

        # Process with agent when there's a new prompt
        if "current_prompt" in st.session_state and st.session_state.current_prompt:
            prompt = st.session_state.current_prompt
            st.session_state.current_prompt = None  # Clear the prompt after processing

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
                        # Use markdown to make text selectable/copyable
                        st.markdown(response_text)

                        # If this is a stakeholder report, offer download option
                        if (
                            any(
                                term in prompt.lower()
                                for term in [
                                    "stakeholder report",
                                    "generate report",
                                    "create report",
                                ]
                            )
                            and len(response_text) > 500
                        ):
                            # Generate a unique key for download button
                            download_key = f"download_report_button_{len(st.session_state.agent_messages)}"

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

                    # Try to reinitialize the agent while preserving conversation history
                    st.warning(
                        "Attempting to reinitialize the agent while preserving conversation..."
                    )

                    # Store the conversation history before recreating the agent
                    conversation_history = (
                        st.session_state.agent_messages.copy()
                        if "agent_messages" in st.session_state
                        else []
                    )

                    if "agent" in st.session_state:
                        del st.session_state.agent

                    new_agent = create_jira_agent(chat_model, tickets)
                    if new_agent:
                        st.session_state.agent = new_agent
                        # Keep the conversation history
                        st.session_state.agent_messages = conversation_history
                        st.success(
                            "Agent reinitialized with conversation history preserved. Please try your question again."
                        )
                    else:
                        st.error(
                            "Could not reinitialize the agent. Please try refreshing the page."
                        )

                # Always rerun to update the UI with new messages
                st.rerun()

        # Get user input at the bottom - ALWAYS after the chat history and responses
        prompt = st.chat_input(
            "Ask about tickets or request a stakeholder report...",
            key="agent_chat_input",
        )

        if prompt:
            # Add user message to chat history
            st.session_state.agent_messages.append({"role": "user", "content": prompt})

            # Store prompt in session state for processing after rerun
            st.session_state.current_prompt = prompt

            # Display user message and rerun to process the message
            with st.chat_message("user"):
                st.markdown(prompt)

            st.rerun()


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
                    "You can now chat about the tickets."
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

            st.session_state.timeout = st.slider(
                "Query Timeout (seconds)", 10, 60, st.session_state.timeout
            )
            st.session_state.max_results = st.number_input(
                "Max Results", 10, 10000, st.session_state.max_results
            )

        # JQL input
        jql_query = st.text_area("Enter JQL Query:", height=100)

        if st.button("Execute JQL"):
            st.session_state.chat_enabled = (
                False  # Reset chat when new query is executed
            )
            if validate_jql(jql_query):
                st.session_state.JQL_QUERY = jql_query
                tickets = load_jira_tickets(
                    jql_query, st.session_state.max_results, load_all=True
                )
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
