#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  GITLAB_TOKEN=<token> bash publish_release.sh <tag> [wheel_file]

Environment:
  GITLAB_TOKEN       GitLab access token with api scope.
  GITLAB_BASE_URL    Optional. Defaults to https://<origin-host>.
  GITLAB_PROJECT_ID  Optional. Defaults to URL-encoded origin project path.
  CURL_CONNECT_TIMEOUT      Optional. Defaults to 15 seconds.
  CURL_MAX_TIME             Optional. Defaults to 3600 seconds.
  RELEASE_DESCRIPTION       Optional. Release description text.
  RELEASE_DESCRIPTION_FILE  Optional. Path to a Markdown file used as the release description.
  REPLACE_PACKAGE           Optional. Set to 1 to delete and re-upload the package for this tag.

Examples:
  GITLAB_TOKEN=glpat-xxx bash publish_release.sh 26.5.3.1
  RELEASE_DESCRIPTION="更新站立策略参数" GITLAB_TOKEN=glpat-xxx bash publish_release.sh 26.5.3.1
  RELEASE_DESCRIPTION_FILE=release.md GITLAB_TOKEN=glpat-xxx bash publish_release.sh 26.5.3.1
  REPLACE_PACKAGE=1 GITLAB_TOKEN=glpat-xxx bash publish_release.sh 26.5.3.1
  GITLAB_TOKEN=glpat-xxx bash publish_release.sh 26.5.3.1 dist/xmigcs-26.5.3.1-*.whl
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

show_response() {
  local title="$1"
  local response_file="$2"

  echo "$title" >&2
  if [[ -s "$response_file" ]]; then
    sed 's/^/  /' "$response_file" >&2
    echo >&2
  fi
}

urlencode() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import quote

print(quote(sys.argv[1], safe=""))
PY
}

json_string() {
  python3 - "$1" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}

json_ids() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

if isinstance(payload, list):
    for item in payload:
        item_id = item.get("id") if isinstance(item, dict) else None
        if item_id is not None:
            print(item_id)
PY
}

json_link_ids_by_url_prefix() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

prefix = sys.argv[2]
if isinstance(payload, list):
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or ""
        item_id = item.get("id")
        if item_id is not None and url.startswith(prefix):
            print(item_id)
PY
}

json_has_link_by_name_or_url() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

target_name = sys.argv[2]
target_url = sys.argv[3]
found = False
if isinstance(payload, list):
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("name") == target_name or item.get("url") == target_url:
            found = True
            break

print("1" if found else "0")
PY
}

parse_origin() {
  local remote_url="$1"

  if [[ "$remote_url" =~ ^git@([^:]+):(.+)$ ]]; then
    ORIGIN_HOST="${BASH_REMATCH[1]}"
    ORIGIN_PROJECT_PATH="${BASH_REMATCH[2]}"
  elif [[ "$remote_url" =~ ^https?://([^/]+)/(.+)$ ]]; then
    ORIGIN_HOST="${BASH_REMATCH[1]}"
    ORIGIN_PROJECT_PATH="${BASH_REMATCH[2]}"
  else
    die "cannot parse remote.origin.url: ${remote_url}"
  fi

  ORIGIN_PROJECT_PATH="${ORIGIN_PROJECT_PATH%.git}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local tag="${1:-}"
  [[ -n "$tag" ]] || {
    usage
    exit 1
  }

  [[ -n "${GITLAB_TOKEN:-}" ]] || die "GITLAB_TOKEN is required"
  command -v git >/dev/null || die "git is required"
  command -v curl >/dev/null || die "curl is required"
  command -v python3 >/dev/null || die "python3 is required"

  local wheel_file="${2:-}"
  if [[ -z "$wheel_file" ]]; then
    shopt -s nullglob
    local wheels=(dist/*.whl)
    shopt -u nullglob
    case "${#wheels[@]}" in
      0) die "no wheel found in dist/. Build it first with: uv build --wheel" ;;
      1) wheel_file="${wheels[0]}" ;;
      *) die "multiple wheels found in dist/. Pass the target wheel explicitly." ;;
    esac
  fi

  [[ -f "$wheel_file" ]] || die "wheel file does not exist: ${wheel_file}"

  local remote_url
  remote_url="$(git config --get remote.origin.url)"
  [[ -n "$remote_url" ]] || die "remote.origin.url is not configured"

  parse_origin "$remote_url"

  local gitlab_base_url="${GITLAB_BASE_URL:-https://${ORIGIN_HOST}}"
  local gitlab_api_url="${gitlab_base_url%/}/api/v4"
  local project_ref
  if [[ -n "${GITLAB_PROJECT_ID:-}" ]]; then
    project_ref="$GITLAB_PROJECT_ID"
  else
    project_ref="$(urlencode "$ORIGIN_PROJECT_PATH")"
  fi

  local encoded_tag
  encoded_tag="$(urlencode "$tag")"
  local curl_connect_timeout="${CURL_CONNECT_TIMEOUT:-15}"
  local curl_max_time="${CURL_MAX_TIME:-3600}"

  local release_description="Release ${tag}"
  local release_description_provided=0
  if [[ -n "${RELEASE_DESCRIPTION_FILE:-}" ]]; then
    [[ -f "$RELEASE_DESCRIPTION_FILE" ]] || die "release description file does not exist: ${RELEASE_DESCRIPTION_FILE}"
    release_description="$(<"$RELEASE_DESCRIPTION_FILE")"
    release_description_provided=1
  elif [[ -n "${RELEASE_DESCRIPTION:-}" ]]; then
    release_description="$RELEASE_DESCRIPTION"
    release_description_provided=1
  fi

  if ! git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    git tag "$tag"
  fi

  git push origin "refs/tags/${tag}"

  local wheel_name package_name package_version encoded_package_name encoded_package_version encoded_wheel_name package_prefix package_url
  wheel_name="$(basename "$wheel_file")"
  package_name="${ORIGIN_PROJECT_PATH##*/}"
  package_version="$(printf '%s' "$tag" | tr '/' '-')"
  encoded_package_name="$(urlencode "$package_name")"
  encoded_package_version="$(urlencode "$package_version")"
  encoded_wheel_name="$(urlencode "$wheel_name")"
  package_prefix="${gitlab_api_url}/projects/${project_ref}/packages/generic/${encoded_package_name}/${encoded_package_version}/"
  package_url="${package_prefix}${encoded_wheel_name}"

  local release_status
  echo "Checking release ${tag}..."
  release_status="$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --connect-timeout "$curl_connect_timeout" \
    --max-time "$curl_max_time" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}")"

  if [[ "${REPLACE_PACKAGE:-}" == "1" ]]; then
    local packages_response packages_status packages_url package_id delete_package_status
    packages_response="$(mktemp)"
    packages_url="${gitlab_api_url}/projects/${project_ref}/packages?package_type=generic&package_name=${encoded_package_name}&package_version=${encoded_package_version}&per_page=100"
    packages_status="$(curl --silent --show-error \
      --output "$packages_response" \
      --write-out "%{http_code}" \
      --connect-timeout "$curl_connect_timeout" \
      --max-time "$curl_max_time" \
      --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
      "$packages_url")"

    if [[ ! "$packages_status" =~ ^2 ]]; then
      show_response "Package lookup failed with HTTP ${packages_status}:" "$packages_response"
      rm -f "$packages_response"
      exit 1
    fi

    while IFS= read -r package_id; do
      delete_package_status="$(curl --silent --output /dev/null --write-out "%{http_code}" \
        --connect-timeout "$curl_connect_timeout" \
        --max-time "$curl_max_time" \
        --request DELETE \
        --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        "${gitlab_api_url}/projects/${project_ref}/packages/${package_id}")"
      if [[ ! "$delete_package_status" =~ ^2 && "$delete_package_status" != "404" ]]; then
        rm -f "$packages_response"
        die "failed to delete existing package ${package_id}, HTTP ${delete_package_status}"
      fi
      echo "Deleted existing package ${package_id}"
    done < <(json_ids "$packages_response")
    rm -f "$packages_response"

    if [[ "$release_status" == "200" ]]; then
      local links_response links_status link_id delete_link_status
      links_response="$(mktemp)"
      links_status="$(curl --silent --show-error \
        --output "$links_response" \
        --write-out "%{http_code}" \
        --connect-timeout "$curl_connect_timeout" \
        --max-time "$curl_max_time" \
        --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}/assets/links")"

      if [[ "$links_status" =~ ^2 ]]; then
        while IFS= read -r link_id; do
          delete_link_status="$(curl --silent --output /dev/null --write-out "%{http_code}" \
            --connect-timeout "$curl_connect_timeout" \
            --max-time "$curl_max_time" \
            --request DELETE \
            --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
            "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}/assets/links/${link_id}")"
          if [[ ! "$delete_link_status" =~ ^2 && "$delete_link_status" != "404" ]]; then
            rm -f "$links_response"
            die "failed to delete existing release asset link ${link_id}, HTTP ${delete_link_status}"
          fi
          echo "Deleted existing release asset link ${link_id}"
        done < <(json_link_ids_by_url_prefix "$links_response" "$package_prefix")
      else
        show_response "Release asset link lookup failed with HTTP ${links_status}:" "$links_response"
      fi
      rm -f "$links_response"
    fi
  fi

  local existing_package_response existing_package_status package_exists
  echo "Checking package ${wheel_name}..."
  existing_package_response="$(mktemp)"
  existing_package_status="$(curl --silent --show-error \
    --output "$existing_package_response" \
    --write-out "%{http_code}" \
    --connect-timeout "$curl_connect_timeout" \
    --max-time "$curl_max_time" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "${gitlab_api_url}/projects/${project_ref}/packages?package_type=generic&package_name=${encoded_package_name}&package_version=${encoded_package_version}&per_page=100")"

  if [[ ! "$existing_package_status" =~ ^2 ]]; then
    show_response "Package lookup failed with HTTP ${existing_package_status}:" "$existing_package_response"
    rm -f "$existing_package_response"
    exit 1
  fi

  package_exists=0
  if [[ -n "$(json_ids "$existing_package_response")" ]]; then
    package_exists=1
  fi
  rm -f "$existing_package_response"

  if [[ "$package_exists" == "1" ]]; then
    echo "Package already exists, reusing it: ${wheel_name}"
  else
    local upload_response upload_status upload_curl_exit
    upload_response="$(mktemp)"
    echo "Uploading package ${wheel_name}..."
    set +e
    upload_status="$(curl --show-error --progress-bar \
      --output "$upload_response" \
      --write-out "%{http_code}" \
      --connect-timeout "$curl_connect_timeout" \
      --max-time "$curl_max_time" \
      --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
      --upload-file "$wheel_file" \
      "$package_url")"
    upload_curl_exit=$?
    set -e

    if [[ "$upload_curl_exit" != "0" ]]; then
      show_response "Package upload failed with curl exit code ${upload_curl_exit}, HTTP ${upload_status:-unknown}:" "$upload_response"
      rm -f "$upload_response"
      exit 1
    fi

    if [[ ! "$upload_status" =~ ^2 ]]; then
      show_response "Package upload failed with HTTP ${upload_status}:" "$upload_response"
      rm -f "$upload_response"
      exit 1
    fi
    rm -f "$upload_response"
  fi

  if [[ "$release_status" == "200" ]]; then
    local existing_links_response existing_links_status link_exists
    existing_links_response="$(mktemp)"
    existing_links_status="$(curl --silent --show-error \
      --output "$existing_links_response" \
      --write-out "%{http_code}" \
      --connect-timeout "$curl_connect_timeout" \
      --max-time "$curl_max_time" \
      --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
      "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}/assets/links")"

    link_exists=0
    if [[ "$existing_links_status" =~ ^2 ]]; then
      link_exists="$(json_has_link_by_name_or_url "$existing_links_response" "$wheel_name" "$package_url")"
    else
      show_response "Release asset link lookup failed with HTTP ${existing_links_status}, trying to create it anyway:" "$existing_links_response"
    fi
    rm -f "$existing_links_response"

    if [[ "$link_exists" == "1" ]]; then
      echo "Release asset link already exists, reusing it: ${wheel_name}"
    else
      local link_response link_status
      link_response="$(mktemp)"
      link_status="$(curl --silent --show-error \
        --output "$link_response" \
        --write-out "%{http_code}" \
        --connect-timeout "$curl_connect_timeout" \
        --max-time "$curl_max_time" \
        --request POST \
        --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        --data-urlencode "name=${wheel_name}" \
        --data-urlencode "url=${package_url}" \
        --data "link_type=package" \
        "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}/assets/links")"

      if [[ ! "$link_status" =~ ^2 ]]; then
        if [[ "$link_status" == "400" || "$link_status" == "409" ]]; then
          show_response "Release asset link may already exist, continuing. GitLab returned HTTP ${link_status}:" "$link_response"
        else
          show_response "Release asset link failed with HTTP ${link_status}:" "$link_response"
          rm -f "$link_response"
          exit 1
        fi
      fi
      rm -f "$link_response"
    fi

    if [[ "$release_description_provided" == "1" ]]; then
      local update_response update_status
      update_response="$(mktemp)"
      update_status="$(curl --silent --show-error \
        --output "$update_response" \
        --write-out "%{http_code}" \
        --connect-timeout "$curl_connect_timeout" \
        --max-time "$curl_max_time" \
        --request PUT \
        --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        --data-urlencode "description=${release_description}" \
        "${gitlab_api_url}/projects/${project_ref}/releases/${encoded_tag}")"
      if [[ ! "$update_status" =~ ^2 ]]; then
        show_response "Release description update failed with HTTP ${update_status}:" "$update_response"
        rm -f "$update_response"
        exit 1
      fi
      rm -f "$update_response"
      echo "Updated release description"
    fi
  else
    local release_payload
    release_payload="$(mktemp)"
    cat > "$release_payload" <<EOF
{
  "name": $(json_string "Release ${tag}"),
  "tag_name": $(json_string "$tag"),
  "description": $(json_string "$release_description"),
  "assets": {
    "links": [
      {
        "name": $(json_string "$wheel_name"),
        "url": $(json_string "$package_url"),
        "link_type": "package"
      }
    ]
  }
}
EOF
    local create_response create_status
    create_response="$(mktemp)"
    create_status="$(curl --silent --show-error \
      --output "$create_response" \
      --write-out "%{http_code}" \
      --connect-timeout "$curl_connect_timeout" \
      --max-time "$curl_max_time" \
      --request POST \
      --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
      --header "Content-Type: application/json" \
      --data @"$release_payload" \
      "${gitlab_api_url}/projects/${project_ref}/releases")"
    rm -f "$release_payload"
    if [[ ! "$create_status" =~ ^2 ]]; then
      show_response "Release creation failed with HTTP ${create_status}:" "$create_response"
      rm -f "$create_response"
      exit 1
    fi
    rm -f "$create_response"
  fi

  echo "Release ${tag} is ready:"
  echo "${gitlab_base_url%/}/${ORIGIN_PROJECT_PATH}/-/releases/${tag}"
}

main "$@"
