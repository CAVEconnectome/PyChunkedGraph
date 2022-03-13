{{/* Generate basic labels */}}
{{- define "chunkedgraph-ingest.labels" }}
  labels:
    generator: helm
    date: {{ now | htmlDate }}
    chart: {{ .Chart.Name }}
    version: {{ .Chart.Version }}
{{- end }}


{{/* Generate volumes */}}
{{- define "chunkedgraph-ingest.volumes" }}
      volumes:
      {{- range $path, $_ :=  .Files.Glob  "secrets/*.json" }}
      {{- $name := $path | base | regexFind "^([^.]+)" }}
        - name: {{ $name | quote }}
          secret:
            secretName: {{ $name | quote }}
      {{- end }}
      {{- range $path, $_ :=  .Files.Glob  "config/*.yml" }}
      {{- $name := $path | base | regexFind "^([^.]+)" }}
        - name: {{ $name | quote }}
          configMap:
            name: {{ $name | quote }}
      {{- end }}
{{- end }}


{{/* Generate volume mounts */}}
{{- define "chunkedgraph-ingest.volume-mounts" }}
          volumeMounts:
          {{- range $path, $_ :=  .Files.Glob  "secrets/*.json" }}
          {{- $fname := $path | base }}
          {{- $name := $path | base | regexFind "^([^.]+)" }}
            - name: {{ $name | quote }}
              mountPath: /root/.cloudvolume/secrets/{{ $fname }}
              subPath: {{ $fname | quote }}
              readOnly: true
          {{- end }}
          {{- range $path, $_ :=  .Files.Glob  "config/*.yml" }}
          {{- $fname := $path | base }}
          {{- $name := $path | base | regexFind "^([^.]+)" }}
            - name: {{ $name | quote }}
              mountPath: /app/config/{{ $fname }}
              subPath: {{ $fname | quote }}
          {{- end }}
{{- end }}