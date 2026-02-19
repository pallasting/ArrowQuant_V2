# Requirements Document: Hybrid Model Architecture

## Introduction

This document specifies the requirements for implementing a hybrid model architecture in OpenClaw that combines local model deployment with cloud API services in a multi-agent collaborative system. The architecture enables intelligent task routing between a fast local model and specialized cloud-based agents, optimizing for both response latency and task complexity.

## Glossary

- **Local_Model**: A locally deployed AI model (MiniCPM-o 4.5) running via vLLM with OpenAI-compatible API
- **Cloud_Agent**: An AI agent using cloud-based API services (Claude, GPT-5.3-Codex)
- **Router_Agent**: The local-fast agent that serves as the entry point and routes tasks to specialists
- **vLLM_Service**: The vLLM server providing OpenAI-compatible endpoints for the local model
- **Agent_Workspace**: An isolated directory containing agent-specific configuration and context
- **Session_Tool**: OpenClaw's inter-agent communication mechanism (sessions_send, sessions_spawn)
- **Model_Provider**: A configured model endpoint in OpenClaw (local, anthropic, openai-codex)
- **Task_Routing**: The process of determining which agent should handle a specific request

## Requirements

### Requirement 1: Local Model Deployment

**User Story:** As a system administrator, I want to deploy a local AI model via vLLM, so that I can provide fast responses for routine tasks without cloud API costs.

#### Acceptance Criteria

1. THE vLLM_Service SHALL expose an OpenAI-compatible API endpoint at /v1/chat/completions
2. WHEN the vLLM_Service receives a chat completion request, THE vLLM_Service SHALL return a response within 5 seconds for typical queries
3. THE vLLM_Service SHALL support streaming responses using Server-Sent Events
4. THE vLLM_Service SHALL support function calling compatible with OpenAI's tool format
5. THE vLLM_Service SHALL load the MiniCPM-o 4.5 model with appropriate quantization for available GPU memory
6. WHEN the vLLM_Service starts, THE vLLM_Service SHALL validate CUDA 11.8+ availability
7. THE vLLM_Service SHALL bind to a configurable host and port (default: localhost:8000)

### Requirement 2: Model Provider Configuration

**User Story:** As a developer, I want to configure the local model as a custom provider in OpenClaw, so that agents can use it like any other model.

#### Acceptance Criteria

1. THE OpenClaw_Config SHALL define a custom model provider named "local" with baseUrl pointing to the vLLM_Service
2. THE Local_Provider SHALL specify the model ID as "minicpm-o-4.5"
3. THE Local_Provider SHALL use the "openai-completions" API type
4. THE Local_Provider SHALL define cost metrics (input: 0, output: 0) to reflect zero API costs
5. THE Local_Provider SHALL specify contextWindow and maxTokens based on MiniCPM-o 4.5 specifications
6. WHEN OpenClaw resolves the "local/minicpm-o-4.5" model reference, THE OpenClaw_Config SHALL return the local provider configuration

### Requirement 3: Multi-Agent Configuration

**User Story:** As a system architect, I want to configure three specialized agents with distinct roles, so that tasks can be routed to the most appropriate model.

#### Acceptance Criteria

1. THE OpenClaw_Config SHALL define an agent with id "local-fast" using model "local/minicpm-o-4.5"
2. THE OpenClaw_Config SHALL define an agent with id "cloud-reasoning" using model "anthropic/claude-opus-4-6"
3. THE OpenClaw_Config SHALL define an agent with id "code-specialist" using model "openai-codex/gpt-5.3-codex"
4. WHEN no agent is specified in a request, THE OpenClaw_Config SHALL default to the "local-fast" agent
5. THE Local_Fast_Agent SHALL have access to session tools (sessions_send, sessions_spawn)
6. THE Cloud_Reasoning_Agent SHALL have a workspace directory at ~/.openclaw/agents/cloud-reasoning/
7. THE Code_Specialist_Agent SHALL have a workspace directory at ~/.openclaw/agents/code-specialist/

### Requirement 4: Agent Workspace Isolation

**User Story:** As a developer, I want each agent to have its own workspace with custom configuration, so that agents can have specialized behaviors and context.

#### Acceptance Criteria

1. WHEN an agent is initialized, THE OpenClaw_System SHALL create a workspace directory for that agent if it does not exist
2. THE Agent_Workspace SHALL contain an AGENTS.md file defining the agent's role and behavior
3. THE Local_Fast_Agent workspace SHALL include routing logic documentation in AGENTS.md
4. THE Cloud_Reasoning_Agent workspace SHALL include deep analysis guidelines in AGENTS.md
5. THE Code_Specialist_Agent workspace SHALL include coding standards and patterns in AGENTS.md
6. WHEN an agent processes a message, THE OpenClaw_System SHALL load the agent's AGENTS.md as context

### Requirement 5: Task Routing Logic

**User Story:** As an end user, I want my requests to be automatically routed to the appropriate agent, so that I get optimal responses without manual agent selection.

#### Acceptance Criteria

1. WHEN the Local_Fast_Agent receives a request, THE Local_Fast_Agent SHALL analyze the task complexity and type
2. WHEN a task requires deep reasoning or complex analysis, THE Local_Fast_Agent SHALL route it to the Cloud_Reasoning_Agent using sessions_send
3. WHEN a task involves code generation or algorithm implementation, THE Local_Fast_Agent SHALL route it to the Code_Specialist_Agent using sessions_send
4. WHEN a task is routine conversation or simple information retrieval, THE Local_Fast_Agent SHALL handle it directly
5. WHEN routing to a specialist agent, THE Local_Fast_Agent SHALL include the original user message and relevant context
6. WHEN a specialist agent completes a task, THE Local_Fast_Agent SHALL receive the response and integrate it into the final reply
7. THE Local_Fast_Agent SHALL return a unified response to the user combining its own analysis and specialist input

### Requirement 6: Direct Agent Invocation

**User Story:** As a power user, I want to directly invoke specific specialist agents, so that I can bypass routing for known task types.

#### Acceptance Criteria

1. WHEN a user specifies agent="cloud-reasoning" in a request, THE OpenClaw_System SHALL route directly to the Cloud_Reasoning_Agent
2. WHEN a user specifies agent="code-specialist" in a request, THE OpenClaw_System SHALL route directly to the Code_Specialist_Agent
3. WHEN a user specifies agent="local-fast" in a request, THE OpenClaw_System SHALL route directly to the Local_Fast_Agent
4. WHEN direct invocation is used, THE OpenClaw_System SHALL bypass the routing logic
5. THE OpenClaw_CLI SHALL support the --agent flag for direct agent selection

### Requirement 7: Inter-Agent Communication

**User Story:** As a system developer, I want agents to communicate using OpenClaw's session tools, so that task delegation is seamless and traceable.

#### Acceptance Criteria

1. WHEN the Local_Fast_Agent delegates a task, THE Local_Fast_Agent SHALL use the sessions_send tool with the target agent ID
2. THE sessions_send tool SHALL create a new session for the specialist agent with the delegated message
3. WHEN a specialist agent completes processing, THE sessions_send response SHALL return the agent's output to the caller
4. THE OpenClaw_System SHALL maintain session isolation between delegated tasks
5. WHEN using sessions_spawn, THE OpenClaw_System SHALL create a sub-agent session under the specified agent ID
6. THE Session_Tool SHALL support passing context and parameters to the target agent

### Requirement 8: Performance Monitoring

**User Story:** As a system administrator, I want to monitor agent usage and performance, so that I can optimize the hybrid architecture.

#### Acceptance Criteria

1. THE Monitoring_System SHALL track the number of requests handled by each agent
2. THE Monitoring_System SHALL record response times for each agent
3. THE Monitoring_System SHALL calculate estimated costs based on model usage and token counts
4. THE Monitoring_System SHALL distinguish between local model usage (zero cost) and cloud API usage
5. THE Monitoring_System SHALL provide a summary report showing agent distribution and cost breakdown
6. WHEN a monitoring script is executed, THE Monitoring_System SHALL output statistics in a human-readable format
7. THE Monitoring_System SHALL support filtering statistics by time range and agent ID

### Requirement 9: Configuration Validation

**User Story:** As a developer, I want the system to validate configuration on startup, so that misconfigurations are caught early.

#### Acceptance Criteria

1. WHEN OpenClaw starts, THE Configuration_Validator SHALL verify that all referenced model providers are defined
2. WHEN OpenClaw starts, THE Configuration_Validator SHALL verify that agent workspace directories exist or can be created
3. WHEN the local model provider is configured, THE Configuration_Validator SHALL verify connectivity to the vLLM_Service
4. IF the vLLM_Service is unreachable, THEN THE Configuration_Validator SHALL log a warning and mark the local provider as unavailable
5. WHEN an agent references a non-existent model, THE Configuration_Validator SHALL fail with a descriptive error message
6. THE Configuration_Validator SHALL verify that session tools are enabled for agents that require routing capabilities

### Requirement 10: Testing Scenarios

**User Story:** As a QA engineer, I want comprehensive test scenarios to validate the hybrid architecture, so that I can ensure correct behavior.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test for simple queries handled entirely by the Local_Fast_Agent
2. THE Test_Suite SHALL include a test for complex reasoning tasks routed to the Cloud_Reasoning_Agent
3. THE Test_Suite SHALL include a test for coding tasks routed to the Code_Specialist_Agent
4. THE Test_Suite SHALL include a test for direct agent invocation bypassing routing
5. THE Test_Suite SHALL verify that specialist responses are correctly integrated into final replies
6. THE Test_Suite SHALL verify that session isolation is maintained between delegated tasks
7. THE Test_Suite SHALL include performance benchmarks comparing local vs cloud response times

### Requirement 11: Documentation and Setup

**User Story:** As a new user, I want clear documentation and setup scripts, so that I can deploy the hybrid architecture quickly.

#### Acceptance Criteria

1. THE Documentation SHALL include step-by-step instructions for installing and configuring vLLM
2. THE Documentation SHALL include the complete OpenClaw configuration file with all three agents
3. THE Documentation SHALL include example AGENTS.md files for each agent workspace
4. THE Documentation SHALL explain the task routing logic and when each agent is used
5. THE Setup_Script SHALL automate vLLM installation and model download
6. THE Setup_Script SHALL create agent workspace directories and populate AGENTS.md files
7. THE Setup_Script SHALL validate CUDA availability and GPU memory before proceeding
8. THE Documentation SHALL include troubleshooting guidance for common issues
