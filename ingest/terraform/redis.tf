resource "google_redis_instance" "redis" {
  name               = var.common_name
  display_name       = var.common_name
  tier               = "BASIC"
  memory_size_gb     = 1
  region             = var.region
  redis_version      = "REDIS_6_X"
  authorized_network = google_compute_network.vpc.name
}

