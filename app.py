import os
import logging
import streamlit as st
from dotenv import load_dotenv
from jira import JIRA
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from functools import lru_cache

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


def chat_about_tickets(tickets, chat_model):
    """
    Handles chat interactions about the displayed tickets.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful assistant that can discuss Jira tickets and provide insights about them."
            )
        ]

    # Create a container for the chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.messages[1:]:  # Skip the system message
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)

        # Get user input
        if prompt := st.chat_input("Ask me about the tickets...", key="chat_input"):
            # Add user message to chat history
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.chat_message("user").write(prompt)

            # Prepare context about tickets
            ticket_context = "\n".join(
                [
                    f"Ticket {data['Key']}: {data['Summary']} (Status: {data['Status']})"
                    for data in [extract_ticket_data(ticket) for ticket in tickets]
                ]
            )

            # Prepare full prompt with context
            full_prompt = f"Context about current tickets:\n{ticket_context}\n\nUser question: {prompt}"

            try:
                # Get AI response
                response = chat_model.predict(full_prompt)

                # Add AI response to chat history
                st.session_state.messages.append(AIMessage(content=response))
                st.chat_message("assistant").write(response)
            except Exception as e:
                logger.error(f"Error in chat response: {e}")
                st.error("Failed to get response from chat agent")


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
                "Start Chat" if not st.session_state.chat_enabled else "End Chat"
            ):
                st.session_state.chat_enabled = not st.session_state.chat_enabled
                st.rerun()

        if st.session_state.chat_enabled and st.session_state.current_tickets:
            chat_model = initialize_chat_agent()
            if chat_model:
                st.write("You can now chat about the displayed tickets:")
                chat_about_tickets(st.session_state.current_tickets, chat_model)
            else:
                st.error("Failed to initialize chat agent")

    except Exception as e:
        logger.error(f"Error displaying tickets table: {e}")
        st.error("Error displaying tickets table")


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
