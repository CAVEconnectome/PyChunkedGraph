variable "worker_nodes_low" {
  default     = 10
  description = "number of worker_low nodes"
}

variable "worker_nodes_mid" {
  default     = 0
  description = "number of worker_mid nodes"
}

variable "worker_nodes_high" {
  default     = 0
  description = "number of worker_high nodes"
}


locals {
  workers_low  = "workers-low"
  workers_mid  = "workers-mid"
  workers_high = "workers-high"
}


resource "google_container_node_pool" "workers_low" {
  name       = local.workers_low
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.worker_nodes_low

  node_config {
    labels = {
      project = var.common_name
    }

    preemptible  = true
    machine_type = "e2-standard-2"
    tags         = ["${var.common_name}-${local.workers_low}"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

resource "google_container_node_pool" "workers_mid" {
  name       = local.workers_mid
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.worker_nodes_mid

  node_config {
    labels = {
      project = var.common_name
    }

    preemptible  = true
    machine_type = "e2-standard-4"
    tags         = ["${var.common_name}-${local.workers_mid}"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

resource "google_container_node_pool" "workers_high" {
  name       = local.workers_high
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.worker_nodes_high

  node_config {
    labels = {
      project = var.common_name
    }

    preemptible  = true
    machine_type = "e2-standard-8"
    tags         = ["${var.common_name}-${local.workers_high}"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}