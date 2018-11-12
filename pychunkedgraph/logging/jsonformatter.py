from pythonjsonlogger import jsonlogger


class JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        """Remap `log_record`s fields to fluentd-gcp counterparts."""
        super(JsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record["time"] = log_record.get("time", log_record["asctime"])
        log_record["severity"] = log_record.get(
            "severity", log_record["levelname"])
        log_record["source"] = log_record.get("source", log_record["name"])
        del log_record["asctime"]
        del log_record["levelname"]
        del log_record["name"]
