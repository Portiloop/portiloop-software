#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_HOSTNAME="$(hostname 2>/dev/null || true)"

is_default_hostname() {
    case "$1" in
        ""|"ubuntu"|"localhost"|"localhost.localdomain"|"rdkx5"|"rdk-x5")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [[ -n "${PORTILOOP_DEVICE_NAME:-}" ]]; then
    DEVICE_NAME="${PORTILOOP_DEVICE_NAME}"
elif [[ "${PORTILOOP_REGENERATE_NAME:-0}" == "1" ]] || is_default_hostname "${CURRENT_HOSTNAME}"; then
    DEVICE_NAME="$(python3 "${SCRIPT_DIR}/generate_device_name.py")"
else
    DEVICE_NAME="${CURRENT_HOSTNAME}"
fi

if [[ "${CURRENT_HOSTNAME}" != "${DEVICE_NAME}" ]]; then
    if command -v hostnamectl >/dev/null 2>&1; then
        sudo hostnamectl set-hostname "${DEVICE_NAME}"
    else
        echo "${DEVICE_NAME}" | sudo tee /etc/hostname >/dev/null
        sudo hostname "${DEVICE_NAME}"
    fi
    if grep -qE '^127\.0\.1\.1[[:space:]]' /etc/hosts; then
        sudo sed -i -E "s/^127\.0\.1\.1[[:space:]].*/127.0.1.1\t${DEVICE_NAME}/" /etc/hosts
    else
        echo -e "127.0.1.1\t${DEVICE_NAME}" | sudo tee -a /etc/hosts >/dev/null
    fi
fi

echo "${DEVICE_NAME}"
