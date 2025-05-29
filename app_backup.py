import os
import streamlit as st
from dotenv import load_dotenv
from jira import JIRA
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# === Configuration ===
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
ITEMS_PER_PAGE = 10  # Pagination constant
global JQL_QUERY
global PROJECT_KEY
global PROJECT_DISPLAY_NAME

JQL_QUERY = ""
PROJECT_KEY = ""
PROJECT_DISPLAY_NAME = ""

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "tickets_loaded" not in st.session_state:
    st.session_state.tickets_loaded = False
if "ticket_knowledge" not in st.session_state:
    st.session_state.ticket_knowledge = ""


def get_available_projects():
    try:
        jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
        projects = {}
        for project in jira.projects():
            display_name = f"{project.name} [{project.key}]"
            projects[display_name] = {"key": project.key, "name": project.name}
        return projects
    except Exception as e:
        st.error(f"Error fetching projects: {str(e)}")
        return {}


def extract_jql_from_query(query: str) -> str:
    prompt = f"""
    Convert the following natural language query into a Jira JQL query.
    Only return the JQL query without any additional text or explanation.
    
    Query: {query}
    
    Follow these rules:
    1. Use proper Jira JQL syntax
    2. Include relevant fields like project, status, priority, etc.
    3. Use appropriate operators (=, !=, IN, AND, OR, etc.)
    4. Add proper ordering if mentioned (ORDER BY)
    
    JQL:
    """

    try:
        jql = llm.predict(prompt).strip()
        return jql
    except Exception as e:
        st.error(f"Error extracting JQL: {str(e)}")
        return None


def validate_jql(jql: str) -> bool:
    try:
        jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
        jira.search_issues(jql, maxResults=1)
        return True
    except Exception as e:
        st.error(f"Invalid JQL: {str(e)}")
        return False


# === Extract Jira tickets ===
@st.cache_data
def load_tickets():
    try:
        with st.spinner("Loading tickets..."):
            jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
            issues = jira.search_issues(
                JQL_QUERY,
                maxResults=st.session_state.max_results,
                expand="renderedFields",
            )

            if not issues:
                st.warning("No tickets found for the current query.")
                return ""

            FIELD_MAP = {
                "id": "key",
                "summary": "fields.summary",
                "description": "fields.description",
                "comments": "fields.comment.comments",
                "status": "fields.status.name",
                "parent": "fields.parent.key",
                "resolution": "fields.resolution.name",
            }

            def extract_fields(issue):
                result = {}
                for field, path in FIELD_MAP.items():
                    try:
                        val = issue
                        for part in path.split("."):
                            val = (
                                getattr(val, part)
                                if hasattr(val, part)
                                else val.get(part)
                            )
                        if field == "comments":
                            val = "\n".join(c.body for c in val) if val else ""
                        result[field] = str(val) if val else ""
                    except Exception:
                        result[field] = ""
                return result

            ticket_summaries = []
            for issue in issues:
                data = extract_fields(issue)
                summary = "\n".join(f"{k}: {v}" for k, v in data.items())
                ticket_summaries.append(summary)

            st.session_state.ticket_knowledge = "\n\n---\n\n".join(ticket_summaries)

            return st.session_state.ticket_knowledge
    except Exception as e:
        st.error(f"Error loading tickets: {str(e)}")
        return ""


def export_tickets_to_csv():
    try:
        jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
        issues = jira.search_issues(JQL_QUERY, maxResults=st.session_state.max_results)

        data = []
        for issue in issues:
            data.append(
                {
                    "Key": issue.key,
                    "Summary": issue.fields.summary,
                    "Status": issue.fields.status.name,
                    "Priority": issue.fields.priority.name,
                    "Assignee": str(issue.fields.assignee),
                    "Created": issue.fields.created,
                }
            )

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error exporting tickets: {str(e)}")
        return None


def display_paginated_tickets(tickets):
    tickets_list = tickets.split("\n\n---\n\n")
    total_pages = len(tickets_list) // ITEMS_PER_PAGE + 1

    st.markdown(f"Page {st.session_state.page_number} of {total_pages}")

    start = (st.session_state.page_number - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE

    for ticket in tickets_list[start:end]:
        with st.expander(ticket.split("\n")[0]):  # Use first line as header
            st.text(ticket)


def save_to_history(query, response):
    st.session_state.query_history.append(
        {
            "query": query,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


# === LangChain Setup ===
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version="2023-05-15",
    temperature=0,
)


def query_tickets(input: str) -> str:
    prompt = f"""
    You are analyzing Jira project data for "{PROJECT_DISPLAY_NAME}" (Project Key: {PROJECT_KEY}). 
    The following tickets are active or recently updated. Use this data to answer the query below.

    TICKETS:
    {st.session_state.ticket_knowledge}

    QUERY:
    {input}

    Answer:
    """
    return llm.predict(prompt)


tools = [
    Tool(
        name="ProjectTicketAnalyzer",
        func=query_tickets,
        description="Use to analyze Jira project tickets for status, issues, progress.",
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=False,
)

# === Streamlit UI ===
st.title("Jira Project Analyst")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    if "max_results" not in st.session_state:
        st.session_state.max_results = 100
    if "timeout" not in st.session_state:
        st.session_state.timeout = 30
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "page_number" not in st.session_state:
        st.session_state.page_number = 1

    st.session_state.timeout = st.slider("Query Timeout (seconds)", 10, 60, 30)
    st.session_state.max_results = st.number_input("Max Results", 10, 500, 100)
    st.session_state.temperature = st.slider("AI Temperature", 0.0, 1.0, 0.0)

    # Update LLM settings
    llm.temperature = st.session_state.temperature

# Project selection
projects = get_available_projects()
if projects:
    project_display = st.selectbox(
        "Select Project:",
        list(projects.keys()),
        help="Select a project to analyze. Format: Project Name [PROJECT-KEY]",
    )
    if project_display:
        PROJECT_KEY = projects[project_display]["key"]
        PROJECT_DISPLAY_NAME = projects[project_display]["name"]

with st.expander("Project Info"):
    st.markdown(f"**Project:** {project_display}")
    st.markdown(f"**Project Key:** {PROJECT_KEY}")
    st.markdown(f"**Project Name:** {PROJECT_DISPLAY_NAME}")
    st.markdown(f"**Current JQL:** `{JQL_QUERY}`")

# === STEP 1: JQL QUERY SELECTION ===
st.header("Step 1: Select JQL Query")

# Option 1: Convert natural language to JQL
nl_query = st.text_input("Convert natural language to JQL:", key="nl_query")
if nl_query:
    with st.spinner("Extracting JQL..."):
        extracted_jql = extract_jql_from_query(nl_query)
        if extracted_jql:
            st.markdown("**Extracted JQL:**")
            st.code(extracted_jql)
            if st.button("Use This JQL"):                if validate_jql(extracted_jql):
                    JQL_QUERY = extracted_jql
                    load_tickets.clear()
                    ticket_knowledge = load_tickets()
                    st.success("JQL query applied and tickets loaded!")
                    st.rerun()
                else:
                    st.error("Invalid JQL. Please try a different query.")

# Option 2: Custom JQL input
st.markdown("**OR**")
custom_jql = st.text_input("Enter custom JQL directly:", key="custom_jql")
if st.button("Use Custom JQL"):
    if validate_jql(custom_jql):        JQL_QUERY = custom_jql
        load_tickets.clear()
        ticket_knowledge = load_tickets()
        st.success("Custom JQL applied and tickets loaded!")
        st.rerun()
    else:
        st.error("Invalid custom JQL. Please check the syntax.")

# Option 3: Default JQL
st.markdown("**OR**")
if st.button("Use Default JQL"):
    if PROJECT_KEY:
        JQL_QUERY = (
            f"project = {PROJECT_KEY} AND statusCategory != Done ORDER BY priority DESC"
        )
        load_tickets.clear()
        ticket_knowledge = load_tickets()
        st.success("Default JQL applied and tickets loaded!")
        st.experimental_rerun()
    else:
        st.error("Please select a project first.")

# Show current tickets and pagination
if JQL_QUERY:
    with st.expander("Current Tickets Being Analyzed", expanded=False):
        if st.session_state.tickets_loaded:
            display_paginated_tickets(st.session_state.ticket_knowledge)
            # Pagination controls
            st.session_state.page_number = st.number_input("Page", min_value=1, value=1)
        else:
            # Load tickets if they're not already loaded
            ticket_knowledge = load_tickets()
            if ticket_knowledge:
                display_paginated_tickets(ticket_knowledge)
                # Pagination controls
                st.session_state.page_number = st.number_input(
                    "Page", min_value=1, value=1
                )

    # Export functionality
    if st.button("Export to CSV"):
        df = export_tickets_to_csv()
        if df is not None:
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV", csv, "jira_tickets.csv", "text/csv", key="download-csv"
            )

# === STEP 2: CHAT WITH TICKETS ===
if st.session_state.tickets_loaded:
    st.header("Step 2: Ask Questions About Your Tickets")
    query = st.text_input(
        "Ask about the project's progress, issues, or tasks:", key="chat_query"
    )

    if query:
        with st.spinner("Analyzing tickets..."):
            response = agent.run(query)
            st.markdown("**Response:**")
            st.write(response)
            save_to_history(query, response)
else:
    st.info(
        "Please select a JQL query in Step 1 to load tickets before asking questions."
    )

# Query History
with st.expander("Query History"):
    for item in reversed(st.session_state.query_history):
        st.markdown(f"**Query ({item['timestamp']}):** {item['query']}")
        st.markdown(f"**Response:** {item['response']}")
        st.markdown("---")
