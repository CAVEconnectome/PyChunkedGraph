terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.15.0"
    }
  }
}

variable "common_name" {
  description = "common name to identify resources"
}

variable "project_id" {
  description = "project id"
}

variable "region" {
  description = "region"
}

variable "zone" {
  description = "zone"
}

provider "google" {
  credentials = file("../google-secret.json")
  project = var.project_id
  region  = var.region
}
