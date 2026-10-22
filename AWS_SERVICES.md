
# AWS Services Learning Guide (Beginner-Friendly)

A simple guide to the main AWS services, what they do, when to use them, and how they fit together in a real project.

> **Goal:** Learn the purpose of each service first. You do not need to master every AWS service at once. Start with IAM, EC2, S3, CloudWatch, RDS, and networking basics, then move to containers, messaging, and AI services.

---

## Table of Contents

1. [Big Picture: How AWS Works](#big-picture-how-aws-works)
2. [Identity and Access Management](#1-iam--identity-and-access-management)
3. [Compute and Hosting](#2-compute-and-hosting)
4. [Storage](#3-storage)
5. [Databases and Caching](#4-databases-and-caching)
6. [Monitoring and Auditing](#5-monitoring-and-auditing)
7. [Messaging and Notifications](#6-messaging-and-notifications)
8. [Infrastructure as Code](#7-infrastructure-as-code)
9. [Content Delivery and Networking](#8-content-delivery-and-networking)
10. [Serverless Computing](#9-serverless-computing)
11. [AI and Agents](#10-ai-and-agents)
12. [How These Services Work Together](#how-these-services-work-together)
13. [Suggested Learning Roadmap](#suggested-learning-roadmap)
14. [Hands-On Practice Projects](#hands-on-practice-projects)
15. [Quick Revision Table](#quick-revision-table)
16. [Official Learning Resources](#official-learning-resources)

---

## Big Picture: How AWS Works

**Amazon Web Services (AWS)** is a cloud platform. Instead of buying and maintaining your own physical servers, you rent computing power, storage, databases, networking, and other services over the internet.

Think of building an application like running a restaurant:

- **IAM:** decides who is allowed into each area.
- **EC2:** a computer where your application can run.
- **S3:** a storage room for files and images.
- **RDS:** an organized system for related data.
- **CloudWatch:** dashboards and alarms that show whether things are working.
- **CloudTrail:** a record of actions taken in your AWS account.
- **Route 53:** helps people find your application using a domain name.
- **CloudFront:** delivers content to users from locations closer to them.
- **Lambda:** runs a small piece of code when something triggers it.

### A few important terms

- **Region:** a geographical area where AWS has data centers, such as `eu-west-1`.
- **Availability Zone (AZ):** an isolated group of data centers within a Region.
- **Resource:** something you create in AWS, such as a server, database, or bucket.
- **Scalability:** the ability to handle more or fewer users or requests.
- **High availability:** designing a system to keep working when a component fails.
- **Pay-as-you-go:** many AWS services charge based on usage, storage, requests, or running time. Always check pricing before creating resources.

---

## 1. IAM — Identity and Access Management

**Purpose:** Control who can access your AWS account and what they can do.

### Easy explanation

IAM is like a security guard. It determines which people and applications can open which doors.

### Main concepts

- **User:** an identity for a person or workload in IAM. For workforce access, AWS generally recommends federation and IAM Identity Center instead of creating long-term IAM users.
- **Group:** a collection of IAM users with shared permissions.
- **Role:** a set of permissions that a trusted user or AWS service can temporarily assume.
- **Policy:** a JSON document that defines allowed or denied actions on resources.
- **Least privilege:** give each identity only the permissions it needs.
- **MFA:** multi-factor authentication adds another verification step.

### Example

An EC2 application needs to read files from one S3 bucket. Give the EC2 instance an **IAM role** with permission to read only that bucket, rather than storing access keys inside your code.

### Learn and practise

1. Understand users, roles, policies, and MFA.
2. Read a simple IAM policy.
3. Practise granting read-only access to a test S3 bucket.
4. Learn the difference between identity-based and resource-based policies.

**Safety rule:** Never commit AWS access keys, secret keys, passwords, or `.env` files to GitHub. Use IAM roles and secure secret storage where possible.

---

## 2. Compute and Hosting

### 2.1 EC2 — Elastic Compute Cloud

**Purpose:** Rent virtual servers in the cloud.

### Easy explanation

EC2 is like renting a computer that runs in an AWS data center. You choose its operating system, size, storage, and network configuration.

### Common uses

- Hosting a FastAPI backend.
- Running background workers.
- Running custom software that needs control over the operating system.
- Testing Linux server administration.

### Important concepts

- **Instance:** the virtual server.
- **AMI:** a template used to launch an instance.
- **Instance type:** the amount and type of CPU, memory, and networking capacity.
- **EBS:** block storage commonly attached to an EC2 instance.
- **Security group:** a virtual firewall controlling inbound and outbound traffic.
- **SSH:** a common way to connect securely to a Linux instance.

### Example

Run a FastAPI application on an EC2 Linux instance behind a reverse proxy such as Nginx. For a production system, also plan for HTTPS, patching, backups, monitoring, and scaling.

### Learn and practise

1. Launch a small test instance.
2. Connect using the recommended secure method.
3. Configure a limited security group.
4. Deploy a simple API.
5. Stop or terminate test resources when finished.

**Important:** EC2 is flexible, but you manage more of the server yourself than with a fully managed application platform. Charges can apply while resources are running or allocated.

### 2.2 ECS — Elastic Container Service

**Purpose:** Run and manage Docker containers without manually managing every container on a server.

### Easy explanation

Docker packages your application and its dependencies into an image. ECS schedules and runs containers based on the configuration you provide.

### Key concepts

- **Docker image:** the package containing your application and dependencies.
- **Container:** a running instance of an image.
- **Task definition:** a recipe describing containers, resources, environment settings, and logging.
- **Task:** a running instance of a task definition.
- **Service:** keeps the desired number of tasks running and can integrate with load balancing.
- **ECR:** Elastic Container Registry, where you can store private container images.
- **Fargate:** a serverless compute option for running ECS tasks without managing EC2 servers.

### Example

Build a Docker image for your FastAPI backend, push it to ECR, and configure ECS to run it. ECS can replace unhealthy tasks and maintain the desired number of running tasks.

### Learn and practise

1. Dockerize a small FastAPI API.
2. Push its image to ECR.
3. Deploy it to ECS, preferably using Fargate for a beginner lab.
4. View logs in CloudWatch.
5. Learn task health checks, environment configuration, and secrets.

**Remember:** ECR stores images; ECS runs and manages containers.

### 2.3 EKS — Elastic Kubernetes Service

**Purpose:** Run Kubernetes on AWS.

### Easy explanation

Kubernetes is a system for managing containers across a cluster. EKS provides a managed Kubernetes control plane, while you still need to understand and manage parts of the cluster and workloads.

### When to use it

- Your team already uses Kubernetes.
- You need Kubernetes APIs and ecosystem tools.
- Your platform has complex container orchestration requirements.

### Important concepts

- **Cluster:** the Kubernetes environment.
- **Pod:** the smallest standard unit Kubernetes schedules.
- **Deployment:** describes the desired state and number of replicas for an application.
- **Service:** provides stable network access to a set of pods.
- **Node:** compute capacity on which pods run.
- **Namespace:** separates groups of Kubernetes resources.

### ECS vs EKS

- **ECS:** AWS-native container orchestration; often simpler if you only need to run containers on AWS.
- **EKS:** managed Kubernetes; useful when Kubernetes compatibility and its ecosystem matter, but usually has more concepts to learn.

**Beginner advice:** Learn Docker first, then ECS. Learn EKS when you have a clear reason to use Kubernetes.

---

## 3. Storage

### S3 — Simple Storage Service

**Purpose:** Store files and objects, such as images, videos, documents, backups, and logs.

### Easy explanation

S3 is like a very large online storage room. You put files into **buckets**, and each file is stored as an **object** with a key (its name/path).

### Common uses

- Profile pictures and workout videos.
- User-uploaded files.
- Application backups and data exports.
- Static website assets.
- Logs and datasets for analytics or AI workflows.

### Important concepts

- **Bucket:** a container for objects.
- **Object:** a stored file and its metadata.
- **Key:** the object's unique name within a bucket.
- **Versioning:** keeps versions of objects so you can recover from some accidental overwrites or deletions.
- **Lifecycle rules:** transition or expire objects according to rules.
- **Storage classes:** options designed for different access patterns and costs.
- **Bucket policy:** resource-based permissions for a bucket.
- **Presigned URL:** a time-limited URL that can grant access to an object.

### Example

For FitAI Coach, store user profile images in a private S3 bucket. The backend can issue a short-lived presigned URL so an authorized user can upload an image without making the whole bucket public.

### Learn and practise

1. Create a private test bucket.
2. Upload and download a file.
3. Enable versioning in a lab bucket.
4. Practise a least-privilege bucket policy.
5. Learn encryption, lifecycle rules, and presigned URLs.

**Safety rule:** Keep private data private. Do not enable public access unless the use case truly requires it and you understand the risk.

---

## 4. Databases and Caching

### 4.1 RDS — Relational Database Service

**Purpose:** Run a managed relational SQL database.

### Easy explanation

RDS manages much of the routine work of running a database, such as provisioning, backups, and maintenance options. You still design schemas, manage access, and plan for performance and recovery.

### Supported database engines

Common options include PostgreSQL, MySQL, MariaDB, Oracle, and Microsoft SQL Server. Exact engine and feature availability depends on AWS Region and configuration.

### When to use it

Use a relational database when your data has relationships and you need SQL queries, constraints, and transactions.

### Example

FitAI Coach might use PostgreSQL to store users, workout plans, exercise history, and subscriptions. A `users` table can relate to many `workout_sessions` records.

### Learn and practise

1. Learn tables, rows, columns, primary keys, and foreign keys.
2. Practise `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and `JOIN`.
3. Learn transactions and indexes.
4. Connect a FastAPI application to a test RDS database securely.
5. Understand backups, maintenance, and high availability.

**Security:** Avoid exposing a database publicly. Prefer private networking and tightly restricted access.

### 4.2 ElastiCache — Managed Caching

**Purpose:** Run managed in-memory data stores, commonly Redis OSS or Memcached, depending on the service and configuration.

### Easy explanation

A cache keeps frequently needed data in fast memory so your application does not need to repeatedly query the main database.

### Common uses

- Caching frequently read data.
- Session or temporary state storage, where appropriate.
- Rate-limiting counters.
- Short-lived results and coordination patterns.

### Example

If an API repeatedly reads a popular exercise list from PostgreSQL, it may cache the result for a short time. The API checks the cache first and queries the database on a cache miss.

### Important concepts

- **Cache hit:** the requested item is in the cache.
- **Cache miss:** the item is not in the cache, so the application fetches it elsewhere.
- **TTL (time to live):** how long an item should remain cached.
- **Invalidation:** deciding when cached data must be removed or refreshed.

**Important:** A cache is not automatically your permanent source of truth. Plan for expiry, stale data, access control, and cache failures.

### 4.3 DynamoDB — NoSQL Database

You wrote “ADB non sql database”; the AWS service name you most likely mean is **DynamoDB**.

**Purpose:** A managed NoSQL database designed for low-latency access at scale.

### Easy explanation

A relational database often organizes information into related tables. DynamoDB stores items in tables and is designed around the access patterns you plan for.

### Main concepts

- **Table:** a collection of items.
- **Item:** one record.
- **Attribute:** a field on an item.
- **Partition key:** determines how items are distributed and located.
- **Sort key:** optional key used to organize related items and support query patterns.
- **On-demand capacity:** capacity mode that adjusts based on traffic, with charges based on usage.
- **Provisioned capacity:** capacity you configure, with optional auto scaling.

### Example

DynamoDB can store user activity events or application state when the access patterns fit its key-based data model. It is not simply a drop-in replacement for PostgreSQL.

### RDS vs DynamoDB

- Choose **RDS** when relational data, SQL, joins, and relational transactions are central.
- Consider **DynamoDB** when you have clear key-based access patterns and need a managed NoSQL design that can scale significantly.

---

## 5. Monitoring and Auditing

### 5.1 CloudWatch

**Purpose:** Monitor AWS resources and applications through metrics, logs, dashboards, and alarms.

### Easy explanation

CloudWatch is your application's health dashboard. It helps answer questions such as: Is the server overloaded? Are errors increasing? Is a function failing?

### Common uses

- EC2 CPU and network metrics.
- Application logs from services such as ECS and Lambda.
- Dashboards for important metrics.
- Alarms for thresholds or metric anomalies.
- Alerts through supported notification integrations.

### Example

Create an alarm when EC2 CPU stays unusually high or when the number of application errors increases. Send an alarm notification to an appropriate destination, such as an SNS topic.

### Learn and practise

1. Find metrics for a test EC2 instance.
2. Send application logs to CloudWatch Logs.
3. Create a dashboard.
4. Create a test alarm.
5. Learn log retention and the costs of storing and querying logs.

### 5.2 CloudTrail

**Purpose:** Record activity and API events in your AWS account for auditing and investigation.

### Easy explanation

CloudTrail is an activity record. It can help you investigate who or what made an AWS API request, which action was attempted, and when it happened.

### Example

If an S3 bucket policy changes unexpectedly, CloudTrail event history can help you investigate the API activity associated with that change.

### CloudWatch vs CloudTrail

- **CloudWatch:** “Is my application healthy? What errors and metrics are happening?”
- **CloudTrail:** “What AWS action happened, and which identity or service made the request?”

CloudTrail is not automatically a guarantee that every event is retained forever. Configure trails, destinations, retention, and integrity protections to match your auditing requirements.

---

## 6. Messaging and Notifications

### 6.1 SQS — Simple Queue Service

**Purpose:** Let one part of an application send work to another part without requiring both parts to run at the same time.

### Easy explanation

SQS is like a task inbox. One service places a message in the queue; a worker processes it later.

### Common uses

- Background email or report generation.
- Processing uploaded files.
- Retrying work after temporary failures.
- Separating a busy API from slower processing tasks.

### Queue types

- **Standard queue:** designed for very high throughput; messages may be delivered more than once and can arrive out of order.
- **FIFO queue:** preserves order within its message group and supports deduplication features, subject to its documented constraints.

### Important concepts

- **Producer:** sends messages.
- **Consumer/worker:** receives and processes messages.
- **Visibility timeout:** period during which a received message is hidden from other consumers.
- **Dead-letter queue (DLQ):** stores messages that repeatedly fail processing.
- **Idempotency:** processing the same message again should not cause unwanted duplicate effects.

### Example

When a user finishes a workout, the API puts a message in SQS. A background worker processes the message to create a report or trigger follow-up work, instead of making the user wait for all processing in the API request.

### 6.2 SNS — Simple Notification Service

**Purpose:** Publish a message to multiple subscribers.

### Easy explanation

SNS is like a broadcast announcement. A publisher sends one message to a topic, and SNS delivers it to its configured subscribers.

### Common uses

- Sending alerts to multiple destinations.
- Fan-out to multiple SQS queues.
- Application notifications through supported delivery mechanisms.
- Forwarding CloudWatch alarm notifications.

### SNS vs SQS

- **SNS:** distributes or broadcasts messages to subscribers.
- **SQS:** stores messages in a queue until consumers process them.

They can work together: SNS publishes an event to multiple SQS queues so separate services can process it independently.

### 6.3 SES — Simple Email Service

**Purpose:** Send and receive email using AWS email infrastructure, subject to configuration and regional availability.

### Common uses

- Verification emails.
- Password reset emails.
- Workout summaries and receipts.
- Transactional application emails.

### What you need to learn

- Verify a domain or email identity.
- Configure SPF, DKIM, and ideally DMARC for domain authentication.
- Understand sandbox restrictions and production access.
- Handle bounces, complaints, and delivery events.
- Keep email templates and recipient data secure.

**Important:** SES is for email, not push notifications to a mobile device. For mobile push, use a push-notification provider and platform-specific services appropriate to your app.

---

## 7. Infrastructure as Code

### CloudFormation

**Purpose:** Create and manage AWS infrastructure using templates rather than clicking through the console each time.

### Easy explanation

CloudFormation is a repeatable instruction file for building AWS resources. You describe what you want, and CloudFormation creates or updates the resources as a **stack**.

### Why it matters

- Repeat the same environment reliably.
- Review infrastructure changes in Git.
- Reduce manual setup mistakes.
- Create development, test, and production environments consistently.

### Example

A template can define an S3 bucket, an IAM role, and a CloudWatch alarm. A team can review the template changes before deploying them.

### Learn and practise

1. Read a small YAML or JSON template.
2. Create a harmless test stack.
3. Update one resource and review the change set.
4. Learn parameters, outputs, and stack deletion behavior.
5. Never put passwords or long-lived secrets directly in templates.

**Related tool:** AWS CDK lets you define infrastructure using familiar programming languages and synthesizes it into CloudFormation templates.

---

## 8. Content Delivery and Networking

### 8.1 CloudFront

**Purpose:** Deliver websites, images, videos, APIs, and other content through a content delivery network (CDN).

### Easy explanation

CloudFront caches and serves content from edge locations closer to users, which can reduce latency and load on the origin server.

### Common uses

- Delivering images and static app assets.
- Hosting static websites using an appropriate origin.
- Caching public API responses where safe.
- Protecting and accelerating content delivery with supported AWS security services.

### Important concepts

- **Origin:** the source CloudFront fetches content from, such as S3 or an application load balancer.
- **Edge location:** a location that helps deliver content near users.
- **Cache behavior:** rules controlling how content is cached and forwarded.
- **Invalidation:** asks CloudFront to remove selected objects from its cache.

**Important:** CloudFront does not automatically make every API faster. Caching depends on headers, cache policies, request patterns, and whether responses are safe to reuse.

### 8.2 API Gateway

**Purpose:** Create and manage API endpoints that connect clients to backend services.

### Easy explanation

API Gateway is a managed front door for APIs. It can route requests to supported backends and apply features such as authorization, throttling, and request/response handling.

### Common integrations

- Lambda functions.
- HTTP endpoints and services, depending on API type and configuration.
- AWS services through supported integrations.

### Example

A mobile app sends a request to an API endpoint. API Gateway validates and routes the request to a Lambda function, which returns a response.

### API Gateway vs an application server

- **API Gateway:** API entry point and management features.
- **EC2/ECS/FastAPI:** can host the application logic that processes requests.
- **Lambda:** can run function code without you managing a server.

API Gateway is optional; a FastAPI app can also be exposed through other architectures, such as a load balancer and an application service.

### 8.3 Route 53

**Purpose:** Domain registration (where supported), DNS hosting, and DNS-based routing and health-check features.

### Easy explanation

DNS is the internet's address book. Route 53 helps map a name such as `api.example.com` to the appropriate destination.

### Learn and practise

1. Learn DNS records: A, AAAA, CNAME, and TXT.
2. Understand hosted zones and nameservers.
3. Point a test subdomain to an application endpoint.
4. Learn TTL and DNS propagation.
5. Understand routing policies and health checks.

**Important:** DNS records do not themselves provide HTTPS. Configure TLS certificates and HTTPS on the relevant service.

---

## 9. Serverless Computing

### Lambda — AWS Lambda

**Purpose:** Run code in response to events without managing the underlying servers.

### Easy explanation

Instead of keeping a server running and managing it, you upload a function. AWS runs the function when a configured trigger invokes it and manages the underlying compute infrastructure.

### Common triggers and uses

- API Gateway requests.
- S3 object events.
- Scheduled events.
- Queue messages from SQS.
- Lightweight automation and event processing.

### Important concepts

- **Function:** your code and its configuration.
- **Trigger/event source:** what invokes the function.
- **Execution role:** permissions the function receives.
- **Timeout and memory:** resource settings that affect execution.
- **Cold start:** initialization delay that can occur when a new execution environment starts.
- **Concurrency:** how many function executions can run at once.

### Example

A new image arrives in S3. An S3 event triggers Lambda to create a thumbnail or inspect the file metadata.

### Learn and practise

1. Write a small Lambda function.
2. Invoke it with a test event.
3. Connect API Gateway to Lambda.
4. Review logs in CloudWatch.
5. Learn IAM permissions, retries, timeouts, and idempotency.

**Serverless does not mean there are no servers.** It means AWS manages the server infrastructure for you.

---

## 10. AI and Agents

### 10.1 Amazon Bedrock

**Purpose:** Build generative AI applications using supported foundation models through a managed AWS service.

### Easy explanation

Bedrock provides access to supported AI models through AWS APIs. You can use those models in an application without managing the model-serving infrastructure yourself.

### Common uses

- Chatbots and question-answering.
- Summarizing documents.
- Generating text.
- Retrieval-Augmented Generation (RAG), where an application retrieves relevant information and provides it as context to a model.
- AI features that use supported tools and workflows.

### Key concepts

- **Foundation model (FM):** a model trained on broad data that can be adapted to many tasks.
- **Inference:** running a model to generate an output.
- **Prompt:** instructions and context sent to a model.
- **RAG:** retrieves relevant information from a knowledge source and supplies it to the model.
- **Guardrails:** configured controls intended to help filter or constrain model inputs and outputs.
- **Model evaluation:** checking quality, safety, latency, and cost for your use case.

### Example

FitAI Coach could use a supported Bedrock model to explain a workout plan in simple language. Keep exercise and health-related claims grounded in trusted data, evaluate outputs, and make it clear when an answer is AI-generated.

### Learn and practise

1. Learn the difference between a model, prompt, and inference.
2. Explore model access and regional availability.
3. Build a small text-generation demo.
4. Try RAG using a small set of trusted documents.
5. Evaluate response quality, privacy, safety, latency, and cost.

**Important:** Model availability, access requirements, prices, and capabilities vary by Region and model. Check the current AWS documentation before choosing a model.

### 10.2 Amazon Bedrock AgentCore

**Purpose:** Infrastructure and capabilities for building, deploying, connecting, and operating AI agents, with features depending on the AgentCore component and current availability.

### Easy explanation

An AI model usually responds to a prompt. An **agent** can be designed to use tools, maintain context, and complete a multi-step task within defined permissions. AgentCore provides building blocks for running and operating these agentic applications.

### Agent-related concepts to learn

- **Agent:** software that uses a model and a controlled workflow to work toward a goal.
- **Tools:** functions or services the agent is allowed to call.
- **Runtime:** the environment that runs agent code.
- **Identity and permissions:** controls that determine which tools and resources an agent can access.
- **Memory/context:** information retained or retrieved to help with a task.
- **Observability:** traces, logs, and metrics used to understand the agent's behavior.
- **Evaluations and guardrails:** checks that help assess quality and reduce unsafe behavior.

### Example

A workout-planning agent might retrieve a user's saved workout plan, call a permitted scheduling tool, and propose a weekly schedule. The application should validate the plan and require user confirmation before actions with significant consequences.

### Bedrock vs AgentCore

- **Bedrock models:** provide model inference and related generative AI capabilities.
- **AgentCore:** provides infrastructure and capabilities for building and operating agents.
- You do not need AgentCore just to make a basic model API call.

**Safety tip:** Give agents narrowly scoped tools, validate their inputs and outputs, protect private data, and require human confirmation for important actions.

---

## How These Services Work Together

Here is one possible AWS architecture for a mobile app such as FitAI Coach:

1. The mobile app calls an API over HTTPS.
2. **Route 53** resolves the API domain name.
3. **CloudFront** may sit in front of selected content or endpoints when its caching and security setup fits the use case.
4. An API entry point routes requests to the backend, which could run on **ECS/Fargate**, **EC2**, or **Lambda + API Gateway**.
5. The backend uses **RDS** for relational records and **S3** for uploaded files.
6. **ElastiCache** may speed up frequently requested data if caching is useful.
7. The backend sends background tasks to **SQS**; workers process them asynchronously.
8. **SNS** distributes selected events or alarms, while **SES** sends emails.
9. **CloudWatch** collects logs, metrics, and alarms.
10. **CloudTrail** records AWS API activity for auditing.
11. **IAM** roles and policies control access between services.
12. **CloudFormation** can define the infrastructure reproducibly.
13. **Bedrock** can power selected generative AI features; **AgentCore** may be useful if you later build agents.

You would not necessarily use all these services in your first version. Start with the smallest architecture that safely meets your needs.

---

## Suggested Learning Roadmap

A practical order for a beginner is:

### Phase 1 — Cloud foundations

- [ ] Learn Regions, Availability Zones, and shared responsibility.
- [ ] Learn IAM roles, policies, and least privilege.
- [ ] Launch a small EC2 instance and understand security groups.
- [ ] Create a private S3 bucket and practise file operations.
- [ ] Learn basic billing, budgets, and cost alerts.

### Phase 2 — Databases and operations

- [ ] Learn SQL and relational database concepts.
- [ ] Understand RDS backups and private connectivity.
- [ ] Learn CloudWatch metrics, logs, dashboards, and alarms.
- [ ] Learn CloudTrail and AWS API auditing.
- [ ] Learn Route 53 DNS and HTTPS basics.

### Phase 3 — Application deployment

- [ ] Learn Docker images and containers.
- [ ] Store a container image in ECR.
- [ ] Deploy a small service to ECS/Fargate.
- [ ] Learn API Gateway and Lambda basics.
- [ ] Learn CloudFormation templates and stacks.

### Phase 4 — Production patterns

- [ ] Learn SQS, retries, visibility timeout, and dead-letter queues.
- [ ] Learn SNS fan-out and SES email basics.
- [ ] Learn CloudFront caching and invalidation.
- [ ] Explore ElastiCache and DynamoDB when the use case fits.
- [ ] Learn secrets management, backups, scaling, and incident response.

### Phase 5 — AI on AWS

- [ ] Learn foundation models, inference, prompts, and RAG.
- [ ] Build a small Bedrock demo after checking model access and cost.
- [ ] Learn agent tools, permissions, evaluation, and observability.
- [ ] Explore Bedrock AgentCore only when you need agent-specific infrastructure.

---

## Hands-On Practice Projects

### Project 1: Static website

**Services:** S3, CloudFront, Route 53 (optional), IAM.

Build a simple static site, store its assets in S3, and optionally deliver it with CloudFront and a custom domain.

**You will learn:** object storage, access control, CDN caching, and DNS.

### Project 2: FastAPI backend on AWS

**Services:** EC2 *or* ECS/Fargate, ECR (for ECS), CloudWatch, IAM, optionally Route 53.

Deploy a small FastAPI endpoint such as `/health`. Add HTTPS using an appropriate supported architecture and monitor the logs.

**You will learn:** deployment, networking, container basics, permissions, and monitoring.

### Project 3: Database-backed API

**Services:** ECS/EC2/Lambda (choose one architecture), RDS, IAM, CloudWatch.

Build CRUD endpoints for exercises or workout sessions and store records in PostgreSQL.

**You will learn:** SQL, database connectivity, credentials, schema design, and backups.

### Project 4: Background processing

**Services:** SQS, a worker running on ECS or Lambda, CloudWatch, optionally SNS.

When a workout finishes, queue a task to generate a summary. Add retry handling and a dead-letter queue.

**You will learn:** asynchronous processing, retries, duplicate handling, and decoupled services.

### Project 5: AI workout explainer

**Services:** Amazon Bedrock, your backend, IAM, CloudWatch.

Create a feature that explains an existing workout plan in plain language. Evaluate outputs, protect user data, and keep the model from inventing user records or making unsupported health claims.

**You will learn:** model inference, prompt design, permissions, monitoring, and AI evaluation.

---

## Quick Revision Table

| Service | Simple meaning | Main job |
|---|---|---|
| IAM | Security and permissions | Controls who/what can access AWS resources |
| EC2 | Virtual computer | Runs software on a server you manage |
| S3 | File/object storage | Stores files, images, backups, and static assets |
| ECS | AWS container manager | Runs and maintains containers |
| EKS | Managed Kubernetes | Runs Kubernetes workloads |
| ECR | Container image registry | Stores container images |
| CloudWatch | Health dashboard and logs | Metrics, logs, alarms, and dashboards |
| CloudTrail | Activity audit trail | Records AWS API activity |
| RDS | Managed SQL database | Stores relational data |
| ElastiCache | In-memory cache | Speeds up selected reads and temporary data access |
| DynamoDB | Managed NoSQL database | Stores data using key-based access patterns |
| SQS | Task queue | Buffers work for asynchronous processing |
| SNS | Message broadcast | Sends published messages to subscribers |
| SES | Email service | Sends and receives email |
| CloudFormation | Infrastructure templates | Creates and updates AWS resources as stacks |
| CloudFront | Content delivery network | Delivers content through edge locations |
| API Gateway | API front door | Creates and manages API endpoints |
| Route 53 | DNS and domain routing | Maps domain names to destinations |
| Lambda | Serverless functions | Runs code in response to events |
| Amazon Bedrock | Managed generative AI | Provides access to supported foundation models |
| Bedrock AgentCore | Agent infrastructure | Helps build and operate AI agent applications |

---

## Official Learning Resources

Prefer official AWS documentation because features, service names, regional support, and prices can change.

- [AWS Skill Builder](https://skillbuilder.aws/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Free Tier and pricing information](https://aws.amazon.com/free/)
- [IAM documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)
- [Amazon EC2 documentation](https://docs.aws.amazon.com/ec2/)
- [Amazon S3 documentation](https://docs.aws.amazon.com/s3/)
- [Amazon ECS documentation](https://docs.aws.amazon.com/ecs/)
- [Amazon EKS documentation](https://docs.aws.amazon.com/eks/)
- [Amazon RDS documentation](https://docs.aws.amazon.com/rds/)
- [Amazon DynamoDB documentation](https://docs.aws.amazon.com/dynamodb/)
- [Amazon CloudWatch documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS CloudTrail documentation](https://docs.aws.amazon.com/cloudtrail/)
- [Amazon SQS documentation](https://docs.aws.amazon.com/sqs/)
- [Amazon SNS documentation](https://docs.aws.amazon.com/sns/)
- [Amazon SES documentation](https://docs.aws.amazon.com/ses/)
- [AWS CloudFormation documentation](https://docs.aws.amazon.com/cloudformation/)
- [Amazon CloudFront documentation](https://docs.aws.amazon.com/cloudfront/)
- [Amazon API Gateway documentation](https://docs.aws.amazon.com/apigateway/)
- [Amazon Route 53 documentation](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html)
- [AWS Lambda documentation](https://docs.aws.amazon.com/lambda/)
- [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Bedrock AgentCore documentation](https://docs.aws.amazon.com/bedrock-agentcore/)

---

## Final Advice

Do not try to learn every service deeply at the same time. For a backend/mobile developer, a useful first milestone is:

**IAM → S3 → EC2 basics → CloudWatch → RDS → Docker → ECR/ECS → Route 53/HTTPS → SQS → Lambda/API Gateway → AI services.**

Practise in a sandbox account, enable MFA, create a billing budget, use least-privilege permissions, and delete resources you no longer need. Free-tier eligibility and charges vary by account, service, Region, and usage, so verify current pricing before running labs.
