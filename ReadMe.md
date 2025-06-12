# Jira Project Analyst

A Streamlit application that connects to Jira, retrieves ticket data based on JQL queries, and uses Azure OpenAI to analyze and generate comprehensive reports with AI-powered chat capabilities.

## Features

- **JQL Query Execution**: Execute JQL queries against your Jira instance with pagination support
- **Intelligent Data Display**: View ticket data in interactive tabular format
- **AI-Powered Chat Assistant**: Chat with an AI assistant about your tickets using specialized tools
- **Advanced Report Generation**: Generate stakeholder reports, technical summaries, and product documentation
- **Tool-Based Analysis**: Automatic tool selection based on your requests
- **Download Capabilities**: Download reports in Markdown format
- **Query History Management**: Maintain and review query history
- **Conversation Memory**: Persistent chat history with context awareness

## AI Tools & Capabilities

### 📊 **Stakeholder Report Generator**
- Creates comprehensive executive-level reports for management and external stakeholders
- Analyzes project health, progress, risks, and blockers
- Provides executive summaries, milestone tracking, and actionable insights
- **Best for:** Management presentations, client updates, board meetings, project reviews

### 📝 **Ticket Summarizer** 
- Creates concise technical summaries focused on product features and development status
- Analyzes tickets to understand product purpose and key features
- Eliminates redundant information for clarity
- **Best for:** Team updates, technical documentation, product overviews, development status

### 📚 **Product Documentation Generator**
- Creates comprehensive technical documentation for software platforms
- Generates detailed, formal technical documentation with structured sections
- Includes technical architecture, API docs, UI/UX guidelines, and troubleshooting
- **Best for:** Internal technical teams, API documentation, system architecture docs, onboarding

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Jira account with API access
- Azure OpenAI account with deployed models

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/JiraSummarizer.git
   cd JiraSummarizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the project root directory with the following variables:

   ```env
   # Jira Configuration
   JIRA_SERVER=https://your-jira-instance.atlassian.net
   JIRA_USER=your-email@example.com
   JIRA_API_TOKEN=your-jira-api-token

   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your-azure-openai-api-key
   AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com
   AZURE_OPENAI_VERSION=2025-04-14
   AZURE_DEPLOYMENT_NAME=your-main-model-deployment-name
   AZURE_DEPLOYMENT_TINY_NAME=your-smaller-model-deployment-name
   ```

2. **Obtain a Jira API Token:**
   - Log in to https://id.atlassian.com/manage-profile/security/api-tokens
   - Select "Create API token with scopes"
   - Give your token a descriptive name
   - Select an expiration date (1 to 365 days)
   - Select the app you want the token to access (Jira)
   - Select the required scopes (READ permissions for Jira)
   - Click "Create"
   - Copy the token and add it to your `.env` file

3. **Set up Azure OpenAI:**
   - Create an Azure OpenAI resource if you don't have one
   - Deploy TWO models:
     - **Main model** (e.g., GPT-4): For complex analysis and report generation
     - **Smaller model** (e.g., GPT-4-mini): For welcome messages and tool descriptions
   - Get your API key and endpoint from the Azure portal
   - Add these details to your `.env` file

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser (typically http://localhost:8501)

## Using the Application

### 1. **Execute JQL Queries**
   - Enter a valid JQL query in the text area
   - Examples:
     - `project = "YOUR-PROJECT-KEY" AND status != Closed ORDER BY updated DESC`
     - `assignee = currentUser() AND status = "In Progress"`
     - `priority in (High, Highest) AND status != Done`
   - Click "Execute JQL" to retrieve tickets
   - The application supports pagination and will load all matching tickets

### 2. **View Ticket Data**
   - Tickets are displayed in an interactive table format
   - Use the table's built-in controls to navigate through the data
   - Total ticket count is displayed below the table

### 3. **AI-Powered Chat Interface**
   - Click "Start Chat" to enable the AI assistant
   - The assistant automatically selects the appropriate tool based on your request
   - **Natural Language Queries:**
     - *"What tickets are high priority?"*
     - *"Show me blocked items and their impact"*
     - *"Which features are complete vs in progress?"*

### 4. **Generate Reports & Documentation**
   
   **Stakeholder Reports:**
   - *"Generate a stakeholder report"*
   - *"Create a report for management"*
   - *"Prepare a client update"*
   
   **Technical Summaries:**
   - *"Summarize the tickets"*
   - *"Give me a project overview"*
   - *"What's the current status?"*
   
   **Product Documentation:**
   - *"Generate technical documentation"*
   - *"Create product docs"*
   - *"Document the system architecture"*

### 5. **Download and Share Reports**
   - Use the "Download Report" button to save reports as Markdown files
   - Reports are automatically timestamped
   - Continue conversations to refine and improve reports

### 6. **Advanced Features**
   - **Conversation Memory**: The assistant remembers your conversation context
   - **Iterative Refinement**: Ask follow-up questions to improve reports
   - **Query History**: Review previous JQL queries and results
   - **Settings**: Adjust timeout limits and maximum result counts

## Example Interactions

### Quick Analysis
```
User: "What are the main risks in these tickets?"
AI: Analyzes tickets and identifies blocking issues, dependencies, and potential delays
```

### Report Generation
```
User: "Generate a stakeholder report for this week's review"
AI: Creates comprehensive executive summary with project status, achievements, and risks
```

### Technical Documentation
```
User: "Create technical documentation for the API features"
AI: Generates detailed documentation with architecture, endpoints, and implementation details
```

## Settings & Configuration

### Sidebar Settings
- **Clear History**: Reset all chat and query history
- **Query Timeout**: Adjust timeout for JQL queries (10-60 seconds)
- **Max Results**: Set maximum number of tickets to retrieve (10-10,000)

### Performance Optimization
- The application uses pagination to handle large datasets efficiently
- Two Azure OpenAI models optimize cost and performance:
  - Main model for complex analysis
  - Smaller model for simple tasks

## Troubleshooting

- **Authentication Issues**: Ensure your Jira API token is valid and not expired
- **JQL Errors**: Verify your JQL syntax using Jira's query builder
- **API Limits**: Be aware of rate limits for both Jira and Azure OpenAI APIs
- **Token Expiration**: Jira API tokens expire based on your settings (1-365 days)
- **Model Deployment**: Ensure both Azure OpenAI models are properly deployed and accessible
- **Chat Issues**: If chat fails, try clearing history and restarting the chat session

## Technical Requirements

- **Python Packages**: See `requirements.txt` for full list
- **Azure OpenAI**: Requires access to GPT-4 or similar models
- **Jira Permissions**: READ access to relevant Jira projects
- **Network**: Outbound HTTPS access to Jira and Azure endpoints

## License

[MIT License](LICENSE)