common_name         = "chunkedgraph-ingest"
project_id          = "neuromancer-seung-import"
region              = "us-east1"
zone                = "us-east1-b"
preemptible_master  = false
preemptible_workers = true

worker_types        = {
  low = {
    count   = 10
    machine = "e2-standard-2"
  },
  # mid = {
  #   count   = 0
  #   machine = "e2-standard-4"
  # },
  # high = {
  #   count   = 0
  #   machine = "e2-standard-8"
  # },
}