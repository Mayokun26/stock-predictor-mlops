# Infrastructure Configuration Variables
# Copy this to terraform.tfvars and fill in your values

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "betting-mlops"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Application Configuration
variable "openai_api_key" {
  description = "OpenAI API Key"
  type        = string
  sensitive   = true
}

variable "news_api_key" {
  description = "News API Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API Key"
  type        = string
  sensitive   = true
  default     = ""
}

# Infrastructure Sizing
variable "database_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "redis_instance_type" {
  description = "ElastiCache instance type"
  type        = string
  default     = "cache.t3.micro"
}

variable "fargate_cpu" {
  description = "Fargate CPU units"
  type        = number
  default     = 512  # 0.5 vCPU
}

variable "fargate_memory" {
  description = "Fargate memory in MB"
  type        = number
  default     = 1024  # 1 GB
}

variable "app_count" {
  description = "Number of app instances"
  type        = number
  default     = 2
}

# Networking
variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

# Optional Domain
variable "domain_name" {
  description = "Domain name (optional)"
  type        = string
  default     = ""
}

variable "create_domain" {
  description = "Whether to create Route53 hosted zone"
  type        = bool
  default     = false
}