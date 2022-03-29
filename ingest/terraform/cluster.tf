resource "google_container_cluster" "cluster" {
  name                     = var.common_name
  location                 = var.zone
  remove_default_node_pool = true
  initial_node_count       = 1

  network                  = google_compute_network.vpc.name
  subnetwork               = google_compute_subnetwork.subnet.name
  networking_mode          = "VPC_NATIVE"
  enable_shielded_nodes    = false

  release_channel {
    channel = "STABLE"
  }

  ip_allocation_policy {}
  addons_config {
    http_load_balancing {
      disabled = true
    }
  }
}

variable "preemptible_master" {
  type        = bool
  default     = false
  description = "should master be preemptible?"
}

resource "google_container_node_pool" "master" {
  name       = "master"
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = 1

  node_config {
    labels = {
      project = var.common_name
    }

    preemptible  = var.preemptible_master
    machine_type = "e2-small"
    tags         = ["${var.common_name}-master"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}