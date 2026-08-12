#!/bin/bash

cat > metaservice_config.json <<'EOF'
{
  "meta_service_url": "tcp://127.0.0.1:5000",
  "config_store_url": "tcp://127.0.0.1:6000",
  "metrics_url": "http://127.0.0.1:8000"
}
EOF

export PYTHONPATH=/home/hanwlax/test-codes/sglang/python:$PYTHONPATH
python -m sglang.srt.mem_cache.storage.ascend_memcache.start_meta_service --config_path "metaservice_config.json"
