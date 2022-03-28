output "region" {
  value       = var.region
  description = "GCloud Region"
}

output "zone" {
  value       = var.zone
  description = "GCloud Zone"
}

output "project_id" {
  value       = var.project_id
  description = "GCloud Project ID"
}

output "kubernetes_cluster_name" {
  value       = google_container_cluster.cluster.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_context" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.cluster.name} --zone ${var.zone} --project ${var.project_id}"
  description = "GKE Cluster Context"
}

output "redis_host" {
  description = "The IP address of ingest redis instance."
  value       = "${google_redis_instance.redis.host}"
}

