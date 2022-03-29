{{/* Create kubernetes configmap object from env list */}}

{{- define "common.configmap-from-env" }}
{{- range . }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .name }}
  namespace: {{ .namespace | default "default" | quote }}
data:
  {{- range $key, $val := .vars }}
  {{ $key }}: {{ $val | quote }}
  {{- end }}
---
{{- end}}
{{- end }}


{{/* Create kubernetes configmap object from YAML values*/}}

{{- define "common.configmap-from-yaml" }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .name }}
  namespace: {{ .namespace | default "default" | quote }}
data:
  {{- range .files }}
  {{ .name }}: |-
    {{- toYaml .content | nindent 4 }}
  {{- end }}
---
{{- end }}


{{/* Create kubernetes configmap object from file strings*/}}

{{- define "common.configmap-from-file-strings" }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .name }}
  namespace: {{ .namespace | default "default" | quote }}
data:
  {{- range $key, $val := .files }}
  {{ $key }}: |-
{{ $val | indent 4 }}
  {{- end }}
---
{{- end }}
