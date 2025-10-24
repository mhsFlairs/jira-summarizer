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
import requests
import base64
import json

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
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_TINY_NAME = os.getenv("AZURE_DEPLOYMENT_TINY_NAME")

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


def search_jira_issues_v3(jql_query, start_at=0, max_results=50, fields=None, expand=None):
    """
    Search JIRA issues using the new v3 API endpoint.
    
    Args:
        jql_query: JQL query string
        start_at: Starting index for pagination
        max_results: Maximum number of results per page
        fields: List of fields to return
        expand: List of fields to expand
    
    Returns:
        Dict containing search results
    """
    try:
        # Prepare authentication
        auth_string = f"{JIRA_USER}:{JIRA_API_TOKEN}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Use the new v3 enhanced search endpoint
        url = f"{JIRA_SERVER}/rest/api/3/search/jql"
        
        # Define default fields needed by extract_ticket_data
        default_fields = [
            'summary', 'status', 'priority', 'assignee', 'updated', 
            'description', 'comment', 'resolution', 'created', 'issuetype',
            'project', 'creator', 'reporter'
        ]
        
        # Use provided fields or default fields
        request_fields = fields if fields else default_fields
        
        # Prepare query parameters for GET request
        params = {
            'jql': jql_query,
            'maxResults': max_results,
            'fields': ','.join(request_fields) if isinstance(request_fields, list) else request_fields,
            'startAt': start_at  # Always include startAt, even if it's 0
        }
            
        if expand:
            params['expand'] = ','.join(expand) if isinstance(expand, list) else expand
        
        logger.info(f"Making request to: {url}")
        logger.info(f"With params: {params}")
        
        # Make GET request
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API returned {len(result.get('issues', []))} issues")
            logger.info(f"Full response keys: {list(result.keys())}")
            logger.info(f"Response structure: total={result.get('total', 'not found')}, startAt={result.get('startAt', 'not found')}, maxResults={result.get('maxResults', 'not found')}, isLast={result.get('isLast', 'not found')}")
            return result
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
    except Exception as e:
        logger.error(f"Error making API request: {e}")
        raise


class JiraIssueWrapper:
    """
    Wrapper class to make the API response compatible with the existing code
    that expects JIRA issue objects.
    """
    def __init__(self, issue_data):
        self.raw = issue_data
        self.key = issue_data.get('key', '')
        self.id = issue_data.get('id', '')
        self.fields = issue_data.get('fields', {})
        
    def __getattr__(self, name):
        # Delegate attribute access to the raw data
        if name in self.raw:
            return self.raw[name]
        elif name in self.fields:
            return self.fields[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def validate_jql(jql: str) -> bool:
    """
    Validates JQL syntax using the new v3 API.
    """
    if not jql or not jql.strip():
        st.warning("Please enter a valid JQL query")
        logger.warning("Empty JQL query submitted")
        return False

    try:
        logger.info(f"Validating JQL: {jql}")
        # Use the new API to validate with just 1 result
        result = search_jira_issues_v3(jql, max_results=1)
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
    Common function to load JIRA tickets with pagination support using the new v3 API.

    Args:
        jql_query: JQL query string
        max_results: Maximum number of results to return (None for unlimited when load_all=True)
        expanded: Whether to expand fields
        load_all: If True, loads all tickets using pagination
    """
    try:
        logger.info(f"Fetching tickets with JQL: {jql_query}")

        if not load_all:
            # Original behavior - load up to max_results
            expand_params = ["renderedFields"] if expanded else None
            result = search_jira_issues_v3(
                jql_query, max_results=max_results or 50, expand=expand_params
            )
            
            # write to json
            
            with open('jira_issues.json', 'w') as f:
                json.dump(result, f, indent=4)
            
            # Convert to wrapper objects for compatibility
            issues = [JiraIssueWrapper(issue) for issue in result.get('issues', [])]
            logger.info(f"Successfully fetched {len(issues)} tickets")
            # write to json
            
            return issues

        # Pagination logic for loading all tickets
        all_issues = []
        start_at = 0
        page_size = 100  # Jira's recommended page size
        total_issues = None
        next_page_token = None

        # Show progress bar in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            try:
                expand_params = ["renderedFields"] if expanded else None

                # Fetch a page of results - use nextPageToken if available, otherwise startAt
                if next_page_token and start_at > 0:
                    # TODO: Implement nextPageToken support in search_jira_issues_v3
                    # For now, continue with startAt
                    logger.info(f"Using startAt={start_at} (nextPageToken available but not implemented yet)")
                
                result = search_jira_issues_v3(
                    jql_query,
                    start_at=start_at,
                    max_results=page_size,
                    expand=expand_params,
                )

                page_issues = result.get('issues', [])
                current_page_count = len(page_issues)
                
                # If this is the first request, handle total count
                if total_issues is None:
                    # The new v3 API doesn't provide total count, we'll paginate until isLast=True
                    if current_page_count > 0:
                        total_issues = 999999  # Unknown total, will paginate until complete
                        logger.info(f"No total count available from API, will paginate until isLast=True")
                    else:
                        total_issues = 0
                        logger.info("No tickets found in first page")
                        break

                # Check if we have any issues in this page
                if current_page_count == 0:
                    logger.info("No issues in this page, stopping")
                    break
                wrapped_issues = [JiraIssueWrapper(issue) for issue in page_issues]
                
                # Debug: Log the keys of issues we're getting to detect duplicates
                if page_issues:
                    page_keys = [issue.get('key', 'no-key') for issue in page_issues]
                    logger.info(f"Page {len(all_issues) // page_size + 1} issue keys: {page_keys[:5]}{'...' if len(page_keys) > 5 else ''}")
                
                # Check for duplicates before adding
                existing_keys = {issue.key for issue in all_issues}
                new_issues = [issue for issue in wrapped_issues if issue.key not in existing_keys]
                duplicate_count = len(wrapped_issues) - len(new_issues)
                
                if duplicate_count > 0:
                    logger.warning(f"Found {duplicate_count} duplicate issues in this page, skipping them")
                
                all_issues.extend(new_issues)
                logger.info(f"Added {len(new_issues)} new issues, total unique issues so far: {len(all_issues)}")

                # Update progress
                if total_issues == 999999:  # Unknown total case
                    # Show indeterminate progress based on pages fetched
                    progress_value = min((len(all_issues) % 200) / 200.0, 0.9)  # Cycling progress, max 90%
                    status_text.text(f"Loaded {len(all_issues)} tickets (fetching until complete)...")
                    progress_bar.progress(progress_value)
                elif max_results and max_results < total_issues:
                    progress = min(len(all_issues) / max_results, 1.0)
                    status_text.text(f"Loaded {len(all_issues)} of {max_results} tickets (limited)...")
                    progress_bar.progress(progress)
                else:
                    progress = min(len(all_issues) / total_issues, 1.0) if total_issues > 0 else 0
                    status_text.text(f"Loaded {len(all_issues)} of {total_issues} tickets...")
                    progress_bar.progress(progress)

                logger.info(f"Fetched page starting at {start_at}, got {current_page_count} tickets. Total so far: {len(all_issues)}")

                # Check if we've reached the end or our limit
                should_break = False
                
                if max_results and len(all_issues) >= max_results:
                    logger.info(f"Reached max_results limit of {max_results}")
                    should_break = True
                elif result.get('isLast', True):  # Default to True if not present
                    logger.info("API indicates this is the last page (isLast=True)")
                    should_break = True
                elif duplicate_count == current_page_count and current_page_count > 0:
                    logger.warning("All issues in this page were duplicates - API pagination may be broken, stopping")
                    should_break = True
                elif len(new_issues) == 0 and current_page_count > 0:
                    logger.warning("No new issues added from this page, stopping to prevent infinite loop")
                    should_break = True
                
                if should_break:
                    # Update the final total if we had estimated it
                    if total_issues == 999999:
                        total_issues = len(all_issues)
                        logger.info(f"Final total tickets: {total_issues}")
                    break

                # Update pagination parameters for next iteration
                # The new API uses nextPageToken, but we'll continue with startAt for now
                start_at += current_page_count
                next_page_token = result.get('nextPageToken')
                logger.info(f"Next page will start at index {start_at}, nextPageToken: {next_page_token}")

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
        
        with open('jira_issues.json', 'w') as f:
            json.dump(result, f, indent=4)
        
        return all_issues

    except Exception as e:
        logger.error(f"Error loading tickets: {e}")
        raise


def extract_ticket_data(issue, include_details: bool = False):
    """
    Extracts relevant data from a JIRA issue.
    Works with both old JIRA library objects and new API response objects.
    """
    try:
        # Handle both wrapper objects and direct API responses
        if isinstance(issue, JiraIssueWrapper):
            fields = issue.fields
            key = issue.key
        else:
            # Fallback for old JIRA library objects
            fields = issue.fields
            key = issue.key

        # Safe field access with fallbacks
        def safe_get(obj, attr, default=None):
            try:
                value = getattr(obj, attr, default)
                if isinstance(value, dict):
                    return value.get('name', str(value)) if value else default
                return value if value is not None else default
            except:
                return default

        def safe_dict_get(d, key, subkey=None, default=None):
            try:
                if isinstance(d, dict):
                    value = d.get(key, default)
                    if subkey and isinstance(value, dict):
                        return value.get(subkey, default)
                    return value
                else:
                    return getattr(d, key, default) if hasattr(d, key) else default
            except:
                return default

        # Extract basic data
        data = {
            "Key": key,
            "Summary": safe_dict_get(fields, 'summary', default="No summary"),
            "Status": safe_dict_get(fields, 'status', 'name', default="Unknown"),
            "Priority": safe_dict_get(fields, 'priority', 'name', default="Unknown"),
            "Assignee": (
                safe_dict_get(fields, 'assignee', 'displayName', default="Unassigned")
                if safe_dict_get(fields, 'assignee') else "Unassigned"
            ),
            "Updated": str(safe_dict_get(fields, 'updated', default=""))[:10],
        }

        if include_details:
            # Extract description
            description = safe_dict_get(fields, 'description')
            if isinstance(description, dict) and 'content' in description:
                # Handle ADF (Atlassian Document Format)
                description_text = extract_text_from_adf(description)
            else:
                description_text = str(description) if description else None

            # Extract comments
            comments = []
            comment_data = safe_dict_get(fields, 'comment')
            if comment_data:
                if isinstance(comment_data, dict) and 'comments' in comment_data:
                    for comment in comment_data.get('comments', []):
                        comment_body = comment.get('body', '')
                        if isinstance(comment_body, dict) and 'content' in comment_body:
                            # Handle ADF format
                            comment_text = extract_text_from_adf(comment_body)
                        else:
                            comment_text = str(comment_body)
                        if comment_text:
                            comments.append(comment_text)
                elif hasattr(comment_data, 'comments'):
                    # Old format
                    comments = [str(c.body) for c in comment_data.comments if hasattr(c, 'body')]

            data.update({
                "Description": description_text,
                "Comments": comments[:3],  # Limit to first 3 comments
                "Resolution": (
                    safe_dict_get(fields, 'resolution', 'name')
                    if safe_dict_get(fields, 'resolution') else None
                ),
            })

        return data
    except Exception as e:
        logger.error(f"Error extracting ticket data for {getattr(issue, 'key', 'unknown')}: {e}")
        logger.error(f"Issue structure: {type(issue)}")
        if hasattr(issue, 'fields'):
            logger.error(f"Fields available: {list(issue.fields.keys()) if isinstance(issue.fields, dict) else dir(issue.fields)}")
        return None


def extract_text_from_adf(adf_content):
    """
    Extracts plain text from Atlassian Document Format (ADF) content.
    """
    try:
        if not isinstance(adf_content, dict):
            return str(adf_content)
        
        def extract_text_recursive(node):
            text_parts = []
            
            if isinstance(node, dict):
                if node.get('type') == 'text':
                    text_parts.append(node.get('text', ''))
                elif 'content' in node:
                    for child in node['content']:
                        text_parts.append(extract_text_recursive(child))
            elif isinstance(node, list):
                for item in node:
                    text_parts.append(extract_text_recursive(item))
            
            return ' '.join(filter(None, text_parts))
        
        return extract_text_recursive(adf_content).strip()
    except Exception as e:
        logger.error(f"Error extracting text from ADF: {e}")
        return str(adf_content)


def initialize_chat_agent():
    """
    Initializes the chat agent using Azure OpenAI.
    """
    try:
        chat_model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version="2024-12-01-preview",
            temperature=0.7,
        )
        return chat_model
    except Exception as e:
        logger.error(f"Failed to initialize chat agent: {e}")
        return None


def initialize_tiny_chat_agent():
    """
    Initializes a smaller chat agent for welcome messages and tool descriptions.
    """
    try:
        tiny_chat_model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_TINY_NAME,
            api_version="2024-12-01-preview",
            temperature=0.2,
        )
        return tiny_chat_model
    except Exception as e:
        logger.error(f"Failed to initialize tiny chat agent: {e}")
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

        # Store the raw tickets text in session state for copying
        st.session_state.raw_tickets_text = tickets_text

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


def generate_tool_description(tool_name, tool_description, chat_model):
    """
    Uses AI to generate a more user-friendly description of a tool.
    """
    prompt = f"""
    Rewrite the following tool description to be more user-friendly and engaging.
    Make it concise but clear, and include an example of when to use it.
    
    Tool Name: {tool_name}
    Technical Description: {tool_description}
    
    Format the response in a single paragraph, starting with an emoji that represents the tool's function.
    """

    try:
        response = chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating tool description: {e}")
        return f"📌 {tool_description}"  # Fallback to original description


def generate_welcome_message(tools, chat_model):
    """
    Generates a welcome message with AI-enhanced tool descriptions.
    """
    try:
        # Generate enhanced descriptions for each tool
        tool_descriptions = []
        for tool in tools:
            enhanced_description = generate_tool_description(
                tool.name, tool.description, chat_model
            )
            tool_descriptions.append(f"- **{tool.name}**: {enhanced_description}")

        tools_section = "\n\n".join(tool_descriptions)

        welcome_message = f"""
# Welcome to Your JIRA Analysis Assistant! 🚀

I'm your AI-powered assistant specialized in analyzing JIRA tickets and providing valuable insights. Let me show you what I can do for you:

## Available Tools 🛠️
{tools_section}

## Example Queries 💬
Here are some ways you can interact with me:
- "Give me a quick summary of all current tickets"
- "Create a stakeholder report focusing on major achievements"
- "Generate documentation for the new features"
- "What are the main blockers in our current sprint?"

## Pro Tips 💡
- Be specific in your requests for more accurate responses
- Feel free to ask follow-up questions or request modifications
- You can download stakeholder reports as markdown files
- Use natural language - no need for special commands

Ready to begin? Just type your question or request below! 👇
"""
        return welcome_message

    except Exception as e:
        logger.error(f"Error generating welcome message: {e}")
        return generate_fallback_welcome_message(tools)


def generate_fallback_welcome_message(tools):
    """
    Generates a basic welcome message if AI enhancement fails.
    """
    tool_descriptions = "\n".join(
        f"- **{tool.name}**: {tool.description}" for tool in tools
    )

    return f"""
# Welcome to Your JIRA Analysis Assistant! 🚀

I'm here to help you analyze JIRA tickets. Here are my available tools:

{tool_descriptions}

Feel free to ask any questions about your tickets!
"""


# 2. Fix the chat_about_tickets function to remove duplicate buttons
# 2. Fix the chat_about_tickets function to use invoke instead of run
def chat_about_tickets(tickets, chat_model, tiny_chat_model):
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

    # Generate and display welcome message when agent is first created
    if "welcome_displayed" not in st.session_state:
        welcome_message = generate_welcome_message(agent.tools, tiny_chat_model)
        st.markdown(welcome_message)
        st.session_state.welcome_displayed = True

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

        # Add buttons for chat and copy tickets
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button(
                "Start Chat" if not st.session_state.chat_enabled else "End Chat",
                key="toggle_chat_button",
            ):
                st.session_state.chat_enabled = not st.session_state.chat_enabled
                # Reset agent when toggling chat
                if "agent" in st.session_state:
                    del st.session_state.agent
                if "agent_messages" in st.session_state:
                    st.session_state.agent_messages = []
                st.rerun()

        with col2:
            # Create tickets text and add copy button
            if tickets:
                # Generate the formatted tickets text
                ticket_data = [
                    extract_ticket_data(ticket, include_details=True)
                    for ticket in tickets
                ]
                ticket_data = [data for data in ticket_data if data is not None]

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

                # Download button for tickets text
                st.download_button(
                    label="Copy Tickets",
                    data=tickets_text,
                    file_name=f"tickets_export_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    key="copy_tickets_button",
                )

        if st.session_state.chat_enabled and st.session_state.current_tickets:
            chat_model = initialize_chat_agent()
            tiny_chat_model = initialize_tiny_chat_agent()
            if chat_model:
                st.write("You can now chat about the tickets.")
                chat_about_tickets(
                    st.session_state.current_tickets, chat_model, tiny_chat_model
                )
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
