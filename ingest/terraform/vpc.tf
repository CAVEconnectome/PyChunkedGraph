resource "google_compute_network" "vpc" {
  name                            = "chunkedgraph-ingest"
  routing_mode                    = "REGIONAL"
  auto_create_subnetworks         = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${google_compute_network.vpc.name}-${var.region}"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.0.0.0/24"
}