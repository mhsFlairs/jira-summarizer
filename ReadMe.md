# Jira Project Analyst

A Streamlit application that connects to Jira, retrieves ticket data based on JQL queries, and uses Azure OpenAI to analyze and generate stakeholder reports.

## Features

- Execute JQL queries against your Jira instance
- View ticket data in a tabular format
- Chat with an AI assistant about your tickets
- Generate comprehensive stakeholder reports
- Download reports in Markdown format
- Maintain query history

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Jira account with API access
- Azure OpenAI account and API access

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
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the project root directory with the following variables:

   ```
   # Jira Configuration
   JIRA_SERVER=https://your-jira-instance.atlassian.net
   JIRA_USER=your-email@example.com
   JIRA_API_TOKEN=your-jira-api-token

   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your-azure-openai-api-key
   AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com/
   AZURE_DEPLOYMENT_NAME=your-model-deployment-name
   ```

2. Obtain a Jira API Token:
   - Log in to https://id.atlassian.com/manage-profile/security/api-tokens
   - Select "Create API token with scopes"
   - Give your token a descriptive name
   - Select an expiration date (1 to 365 days)
   - Select the app you want the token to access (Jira)
   - Select the required scopes (READ permissions for Jira)
   - Click "Create"
   - Copy the token and add it to your `.env` file

3. Set up Azure OpenAI:
   - Create an Azure OpenAI resource if you don't have one
   - Deploy a model (e.g., GPT-4 or GPT-3.5-turbo)
   - Get your API key and endpoint from the Azure portal
   - Add these details to your `.env` file

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser (typically http://localhost:8501)

## Using the Application

1. **Execute JQL Queries**:
   - Enter a valid JQL query in the text area
   - Examples:
     - `project = "YOUR-PROJECT-KEY" AND status != Closed ORDER BY updated DESC`
     - `assignee = currentUser() AND status = "In Progress"`
   - Click "Execute JQL" to retrieve tickets

2. **View Ticket Data**:
   - Tickets will be displayed in a table format
   - Scroll or use the table's built-in controls to navigate through the data

3. **Chat About Tickets**:
   - Click "Start Chat" to enable the chat interface
   - Ask questions about the tickets or request analysis
   - Examples:
     - "How many tickets are in 'In Progress' status?"
     - "What are the high priority issues?"

4. **Generate Stakeholder Reports**:
   - In the chat, ask for a report:
     - "Generate a stakeholder report"
     - "Create a report for management"
   - The AI will analyze the tickets and create a comprehensive report
   - Use the "Download Report" button to save the report as a Markdown file

5. **Refine Reports**:
   - Continue the conversation to modify the report
   - Examples:
     - "Add more details about the risks"
     - "Focus more on the upcoming milestones"
     - "Make the executive summary more concise"

## Troubleshooting

- **Authentication Issues**: Ensure your Jira API token is valid and has not expired
- **JQL Errors**: Verify your JQL syntax is correct
- **API Limits**: Be aware of rate limits for both Jira and Azure OpenAI APIs
- **Token Expiration**: Remember that Jira API tokens expire based on your settings (1-365 days)

## License

[MIT License](LICENSE)