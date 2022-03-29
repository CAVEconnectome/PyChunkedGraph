{{/* Create kubernetes secret object from strings */}}

{{- define "common.secret-from-file-strings" }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ .name }}
  namespace: {{ .namespace | default "default" | quote }}
type: Opaque
data:
  {{- range $key, $val := .files }}
  {{ $key }}: |-
    {{ $val | b64enc }}
  {{- end }}
---
{{ end -}}