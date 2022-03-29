{{/* Create kubernetes deployment object */}}

{{- define "common.deployment" }}
{{- if .enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .name | quote }}
  namespace: {{ .namespace | default "default" | quote }}
spec:
{{- if not .hpa.enabled }}
  replicas: {{ .replicaCount }}
{{- end }}
  selector:
    matchLabels:
      app: {{ .name | quote }}
  template:
    metadata:
      annotations:
        {{- if .helmRollOnUpgrade }}
        rollme: {{ randAlphaNum 5 | quote }}
        {{- end }}
        {{- range $key, $val := .annotations }}
        {{ $key }}: {{ $val | quote }}
        {{- end }}
      labels:
        app: {{ .name | quote }}
        {{- range $key, $val := .labels }}
        {{ $key }}: {{ $val | quote }}
        {{- end }}
    spec:
      affinity:
        {{- toYaml .affinity | nindent 8 }}
      volumes:
        {{- toYaml .volumes | nindent 8 }}
      containers:
        - name: {{ .name | quote }}
          image: >-
            {{ required "repo" .image.repository }}:{{ required "tag" .image.tag }}
          imagePullPolicy: {{ .image.pullPolicy | quote }}
          ports:
            {{- toYaml .ports | nindent 12 }}
          resources:
          {{- if .resources }}
            {{- toYaml .resources | nindent 12 }}
          {{- end }}
          envFrom:
          {{- if .envFrom }}
            {{- toYaml .envFrom | nindent 12 }}
          {{- end }}
          {{- range .env }}
          - configMapRef:
              name: {{ .name }}
          {{- end }}
          volumeMounts:
          {{- if .volumeMounts }}
            {{- toYaml .volumeMounts | nindent 12 }}
          {{- end }}
          {{- if .command }}
          command:
            {{- toYaml .command | nindent 12 }}
          {{- else }}
          command: [bash, -c, "trap : TERM INT; sleep infinity & wait"]
          {{- end }}
      imagePullSecrets:
        {{- toYaml .imagePullSecrets | nindent 8 }}
      nodeSelector:
        {{- toYaml .nodeSelector | nindent 8 }}
---
{{- end }}
{{- end }}
