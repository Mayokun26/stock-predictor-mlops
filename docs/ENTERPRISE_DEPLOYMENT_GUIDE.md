# Enterprise MLOps Stock Prediction System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Enterprise MLOps Stock Prediction System in production environments. The system is designed for financial institutions requiring high-availability, scalable machine learning infrastructure with comprehensive monitoring and risk management.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web Frontend  │
│     (NGINX)     │────│   (Kong/Istio)  │────│   (React/Vue)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   ML API Pods   │    │  Feature Store  │    │   MLflow    │ │
│  │   (FastAPI)     │    │    (Redis)      │    │  Registry   │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   PostgreSQL    │    │   Prometheus    │    │   Grafana   │ │
│  │   (Primary)     │    │   Monitoring    │    │ Dashboards  │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Alertmanager  │    │   Log Storage   │
│ (Market APIs)   │    │   (PagerDuty)   │    │ (ELK/Loki)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Infrastructure Requirements

**Minimum Production Environment:**
- **Kubernetes Cluster**: 3 master nodes, 6 worker nodes
- **Node Specifications**: 8 CPU cores, 32GB RAM, 500GB SSD per node
- **Network**: 10Gbps internal networking, redundant internet connectivity
- **Storage**: High-performance block storage with snapshots (1TB minimum)

**Recommended Production Environment:**
- **Kubernetes Cluster**: 3 master nodes, 12 worker nodes  
- **Node Specifications**: 16 CPU cores, 64GB RAM, 1TB NVMe SSD per node
- **Network**: 25Gbps internal networking, multiple ISP connections
- **Storage**: Enterprise-grade storage with replication (5TB)

### Software Dependencies

```bash
# Core Infrastructure
- Kubernetes v1.28+
- Docker v24.0+
- Helm v3.12+
- NGINX Ingress Controller
- cert-manager for TLS

# Monitoring Stack
- Prometheus Operator
- Grafana v10.0+
- AlertManager
- Jaeger (distributed tracing)

# Data Stack
- PostgreSQL v15+ (with replication)
- Redis v7+ (with clustering)
- MinIO (S3-compatible storage)

# Security
- Vault (secrets management)
- Istio Service Mesh (optional)
- OPA Gatekeeper (policy enforcement)
```

## Deployment Process

### 1. Cluster Preparation

```bash
# Create namespace
kubectl create namespace mlops-production

# Apply resource quotas
kubectl apply -f k8s/production/resource-quotas.yaml

# Set up RBAC
kubectl apply -f k8s/production/rbac.yaml

# Configure network policies
kubectl apply -f k8s/production/network-policies.yaml
```

### 2. Secrets Management

```bash
# Create secrets for database connections
kubectl create secret generic postgres-credentials \
  --from-literal=username=mlops_user \
  --from-literal=password=<secure-password> \
  --namespace=mlops-production

# Create secrets for external API keys
kubectl create secret generic api-credentials \
  --from-literal=alpha-vantage-key=<api-key> \
  --from-literal=news-api-key=<api-key> \
  --namespace=mlops-production

# Create TLS certificates
kubectl create secret tls mlops-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=mlops-production
```

### 3. Database Deployment

```bash
# Deploy PostgreSQL with replication
helm install postgresql-primary bitnami/postgresql \
  --set auth.postgresPassword=<secure-password> \
  --set auth.database=mlops_production \
  --set architecture=replication \
  --set primary.persistence.size=500Gi \
  --set readReplicas.replicaCount=2 \
  --namespace=mlops-production

# Deploy Redis cluster
helm install redis-cluster bitnami/redis-cluster \
  --set cluster.nodes=6 \
  --set cluster.replicas=1 \
  --set persistence.size=200Gi \
  --namespace=mlops-production
```

### 4. Application Deployment

```bash
# Deploy the main application
helm install mlops-api ./helm/mlops-api \
  --set image.tag=v2.0.0 \
  --set replicaCount=6 \
  --set resources.requests.memory=2Gi \
  --set resources.requests.cpu=1000m \
  --set resources.limits.memory=4Gi \
  --set resources.limits.cpu=2000m \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=6 \
  --set autoscaling.maxReplicas=20 \
  --namespace=mlops-production

# Deploy MLflow
helm install mlflow ./helm/mlflow \
  --set persistence.size=1Ti \
  --set resources.requests.memory=4Gi \
  --namespace=mlops-production
```

### 5. Monitoring Deployment

```bash
# Deploy Prometheus Operator
helm install prometheus-operator prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.size=50Gi \
  --namespace=monitoring

# Apply custom dashboards
kubectl apply -f monitoring/dashboards/ --namespace=monitoring

# Apply alerting rules
kubectl apply -f monitoring/alerts/ --namespace=monitoring
```

### 6. Security Configuration

```bash
# Enable Pod Security Standards
kubectl label namespace mlops-production \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted

# Deploy NetworkPolicies
kubectl apply -f k8s/production/network-policies.yaml

# Configure service mesh (if using Istio)
kubectl label namespace mlops-production istio-injection=enabled
kubectl apply -f k8s/production/istio/
```

## Configuration

### Environment Variables

```yaml
# Production configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-config
  namespace: mlops-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://mlops_user:password@postgresql-primary:5432/mlops_production"
  REDIS_URL: "redis://redis-cluster:6379"
  MLFLOW_TRACKING_URI: "http://mlflow:5000"
  
  # Performance settings
  WORKERS: "4"
  MAX_CONNECTIONS: "1000"
  CONNECTION_POOL_SIZE: "20"
  
  # Feature flags
  ENABLE_A_B_TESTING: "true"
  ENABLE_CACHING: "true"
  ENABLE_CIRCUIT_BREAKERS: "true"
  
  # Security
  CORS_ORIGINS: "https://trading-platform.company.com"
  JWT_SECRET_KEY: "reference:vault:mlops/jwt-secret"
  RATE_LIMIT_ENABLED: "true"
```

### Resource Limits

```yaml
# Production resource configuration
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    ephemeral-storage: "10Gi"
  limits:
    memory: "4Gi"
    cpu: "2000m"
    ephemeral-storage: "20Gi"

# JVM settings for MLflow
env:
- name: JAVA_OPTS
  value: "-Xmx3g -Xms1g -XX:+UseG1GC"
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlops-api-hpa
  namespace: mlops-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlops-api
  minReplicas: 6
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Cluster Autoscaler

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  nodes.max: "50"
  nodes.min: "6"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
  skip-nodes-with-system-pods: "false"
```

## Monitoring and Alerting

### Key Metrics

**Application Metrics:**
- Request rate (requests/second)
- Response latency (p50, p95, p99)
- Error rate (%)
- Prediction accuracy
- Model drift scores
- Feature store hit rate

**Business Metrics:**
- Portfolio return
- Sharpe ratio
- Maximum drawdown
- Trading volume
- Profit/loss

**Infrastructure Metrics:**
- CPU utilization
- Memory usage
- Disk I/O
- Network throughput
- Pod restart count

### SLA Configuration

```yaml
# Service Level Objectives
apiVersion: sloth.slok.dev/v1
kind: PrometheusServiceLevel
metadata:
  name: mlops-api-slo
  namespace: mlops-production
spec:
  service: mlops-api
  labels:
    team: mlops
  slos:
  - name: requests-availability
    objective: 99.9
    description: 99.9% of requests should be successful
    sli:
      events:
        error_query: sum(rate(http_requests_total{code=~"5.."}[5m]))
        total_query: sum(rate(http_requests_total[5m]))
    alerting:
      name: MLOpsAPIRequestsAvailability
      labels:
        severity: critical
      annotations:
        summary: MLOps API requests availability is below SLO
        
  - name: requests-latency
    objective: 95
    description: 95% of requests should complete within 500ms
    sli:
      events:
        error_query: sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
        total_query: sum(rate(http_request_duration_seconds_count[5m]))
    alerting:
      name: MLOpsAPIRequestsLatency
      labels:
        severity: warning
```

### Alert Routing

```yaml
# Alertmanager configuration
route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
  - match:
      severity: critical
      team: mlops
    receiver: 'mlops-pagerduty'
    continue: true
  - match:
      severity: warning
      team: mlops
    receiver: 'mlops-slack'

receivers:
- name: 'mlops-pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: 'MLOps Critical Alert: {{ .GroupLabels.alertname }}'
    
- name: 'mlops-slack'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#mlops-alerts'
    title: 'MLOps Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Security Hardening

### Network Security

```yaml
# Network Policy - Deny all by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: mlops-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Allow API to database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-database
  namespace: mlops-production
spec:
  podSelector:
    matchLabels:
      app: mlops-api
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
```

### Pod Security

```yaml
# Pod Security Context
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001
  seccompProfile:
    type: RuntimeDefault
  
containers:
- name: mlops-api
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
  volumeMounts:
  - name: tmp-volume
    mountPath: /tmp
  - name: cache-volume
    mountPath: /app/cache

volumes:
- name: tmp-volume
  emptyDir: {}
- name: cache-volume
  emptyDir: {}
```

### RBAC Configuration

```yaml
# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlops-api
  namespace: mlops-production

---
# Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlops-api-role
  namespace: mlops-production
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlops-api-binding
  namespace: mlops-production
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: mlops-api-role
subjects:
- kind: ServiceAccount
  name: mlops-api
  namespace: mlops-production
```

## Disaster Recovery

### Backup Strategy

```bash
# Database backups (daily)
#!/bin/bash
BACKUP_DIR="/backups/postgres/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

kubectl exec -n mlops-production postgresql-primary-0 -- \
  pg_dump -U mlops_user -d mlops_production | \
  gzip > $BACKUP_DIR/mlops_production_$(date +%Y%m%d_%H%M%S).sql.gz

# Model artifacts backup (weekly)
kubectl exec -n mlops-production mlflow-0 -- \
  tar -czf /tmp/models_backup_$(date +%Y%m%d).tar.gz /mlflow/artifacts

kubectl cp mlops-production/mlflow-0:/tmp/models_backup_$(date +%Y%m%d).tar.gz \
  /backups/models/

# Configuration backup
kubectl get configmaps -n mlops-production -o yaml > \
  /backups/k8s/configmaps_$(date +%Y%m%d).yaml

kubectl get secrets -n mlops-production -o yaml > \
  /backups/k8s/secrets_$(date +%Y%m%d).yaml
```

### Recovery Procedures

```bash
# Database recovery
kubectl exec -n mlops-production postgresql-primary-0 -- \
  createdb -U postgres mlops_production_restored

zcat /backups/postgres/20240830/mlops_production_backup.sql.gz | \
kubectl exec -i -n mlops-production postgresql-primary-0 -- \
  psql -U mlops_user -d mlops_production_restored

# Model artifacts recovery
kubectl cp /backups/models/models_backup_20240830.tar.gz \
  mlops-production/mlflow-0:/tmp/

kubectl exec -n mlops-production mlflow-0 -- \
  tar -xzf /tmp/models_backup_20240830.tar.gz -C /

# Application recovery
helm upgrade mlops-api ./helm/mlops-api \
  --set image.tag=v2.0.0 \
  --namespace=mlops-production
```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL production settings
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_predictions_symbol_created ON predictions(symbol, created_at);
CREATE INDEX CONCURRENTLY idx_predictions_model_version ON predictions(model_version);
CREATE INDEX CONCURRENTLY idx_features_symbol_updated ON features(symbol, updated_at);

SELECT pg_reload_conf();
```

### Application Optimization

```yaml
# JVM tuning for MLflow
env:
- name: JAVA_OPTS
  value: |
    -Xmx6g
    -Xms2g
    -XX:+UseG1GC
    -XX:MaxGCPauseMillis=200
    -XX:+UseStringDeduplication
    -XX:+OptimizeStringConcat

# FastAPI settings
env:
- name: WORKERS
  value: "8"
- name: MAX_REQUESTS
  value: "1000"
- name: MAX_REQUESTS_JITTER
  value: "100"
- name: PRELOAD
  value: "true"
```

### Cache Optimization

```yaml
# Redis configuration
redis-config: |
  maxmemory 8gb
  maxmemory-policy allkeys-lru
  timeout 300
  tcp-keepalive 300
  save 900 1
  save 300 10
  save 60 10000
```

## Troubleshooting Guide

### Common Issues

**High Memory Usage:**
```bash
# Check memory usage by pod
kubectl top pods -n mlops-production

# Get detailed resource usage
kubectl describe pod <pod-name> -n mlops-production

# Check for memory leaks
kubectl exec -n mlops-production <pod-name> -- ps aux --sort=-%mem
```

**Database Connection Issues:**
```bash
# Check database connectivity
kubectl exec -n mlops-production <api-pod> -- \
  psql -h postgresql-primary -U mlops_user -d mlops_production -c "SELECT 1;"

# Check connection pool status
kubectl logs -n mlops-production <api-pod> | grep "connection pool"

# Restart database connections
kubectl rollout restart deployment/mlops-api -n mlops-production
```

**Model Loading Failures:**
```bash
# Check MLflow connectivity
kubectl exec -n mlops-production <api-pod> -- \
  curl -f http://mlflow:5000/api/2.0/mlflow/experiments/list

# Check model artifacts
kubectl exec -n mlops-production mlflow-0 -- \
  ls -la /mlflow/artifacts/

# Force model reload
kubectl exec -n mlops-production <api-pod> -- \
  python -c "import mlflow; print(mlflow.list_experiments())"
```

### Health Check Commands

```bash
# Overall system health
kubectl get pods -n mlops-production
kubectl get services -n mlops-production
kubectl get ingress -n mlops-production

# Application health
curl -f http://<ingress-url>/health

# Database health
kubectl exec -n mlops-production postgresql-primary-0 -- \
  pg_isready -U mlops_user

# Redis health
kubectl exec -n mlops-production redis-cluster-0 -- \
  redis-cli ping
```

## Compliance and Auditing

### Audit Logging

```yaml
# Enable Kubernetes audit logging
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Request
  namespaces: ["mlops-production"]
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
  - group: "apps"
    resources: ["deployments", "replicasets"]
```

### Data Governance

```yaml
# Data retention policies
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-retention-config
data:
  prediction_data_retention_days: "365"
  log_retention_days: "90"
  model_artifact_retention_days: "1095"  # 3 years
  audit_log_retention_days: "2555"  # 7 years
```

### Security Scanning

```bash
# Container image scanning
trivy image mlops-api:v2.0.0

# Kubernetes cluster scanning
kube-bench run --targets master,node,etcd,policies

# Runtime security monitoring
kubectl apply -f https://raw.githubusercontent.com/falcosecurity/falco/master/falco-daemonset-config.yaml
```

## Migration Guide

### From Development to Production

1. **Environment Variables Update**
2. **Database Migration**
3. **SSL/TLS Configuration**
4. **Monitoring Setup**
5. **Backup Configuration**
6. **Performance Testing**
7. **Security Hardening**
8. **Documentation Update**

### Version Upgrade Process

```bash
# 1. Backup current state
./scripts/backup-production.sh

# 2. Deploy to staging
helm upgrade mlops-api ./helm/mlops-api \
  --set image.tag=v2.1.0 \
  --namespace=mlops-staging

# 3. Run integration tests
./scripts/run-integration-tests.sh staging

# 4. Blue-green deployment to production
./scripts/blue-green-deploy.sh v2.1.0

# 5. Verify deployment
./scripts/verify-production.sh

# 6. Clean up old version
./scripts/cleanup-old-version.sh
```

## Appendix

### Useful Commands

```bash
# Scale application
kubectl scale deployment mlops-api --replicas=10 -n mlops-production

# Update configuration without restart
kubectl patch configmap mlops-config -n mlops-production --patch '{"data":{"LOG_LEVEL":"DEBUG"}}'

# Port forward for debugging
kubectl port-forward service/mlops-api 8080:8000 -n mlops-production

# Exec into pod for debugging
kubectl exec -it deployment/mlops-api -n mlops-production -- /bin/bash

# View logs with filtering
kubectl logs -f deployment/mlops-api -n mlops-production | grep ERROR
```

### Resource Planning

**Small Deployment (Development/Testing):**
- 3 nodes, 4 CPU, 16GB RAM each
- 50-100 requests/second
- 1 week data retention

**Medium Deployment (Staging/Small Production):**
- 6 nodes, 8 CPU, 32GB RAM each  
- 500-1000 requests/second
- 3 months data retention

**Large Deployment (Production):**
- 12+ nodes, 16 CPU, 64GB RAM each
- 5000+ requests/second
- 1+ year data retention

---

For additional support, contact the MLOps Engineering Team at mlops-support@company.com