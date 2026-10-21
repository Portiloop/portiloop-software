#!/bin/bash

# Updates the Portiloop software of a Portiloop that is already installed (do not run `make` again: it reinstalls
# everything from scratch). Run it on the Portiloop, as mendel: bash ~/portiloop-software/update.sh
#
# 1. Checks that the Portiloop is online, and offers nmtui to connect it to a Wi-Fi network otherwise.
# 2. Checks for local modifications, and offers to stash them.
# 3. Pulls the latest version of the software.
# 4. Installs the system files (services, limits, udev rules) that changed, and offers to reboot.
#
# Everything is in functions called at the very end, as `git pull` may modify this file while it runs. After the pull,
# the new version of this script installs the system files (--after-pull), so that new system files are installed too.

REPO_URL_CHECK="https://github.com"
SETUP_DIR="portiloop/setup_files"
PIP="$HOME/miniforge3/envs/portiloop/bin/pip3"

# System files installed by `make` (keep in sync with step5.temp in the Makefile): "source destination"
# The Jupyter service is added by jupyter_service_source, as it depends on the installation.
SYSTEM_FILES=(
    "asound.conf /etc/asound.conf"
    "create_login_folder.service /etc/systemd/system/create_login_folder.service"
    "simplegui.service /etc/systemd/system/simplegui.service"
    "limits.conf /etc/security/limits.d/portiloop.conf"
    "sd_card.rules /etc/udev/rules.d/99-portiloop-sd-card.rules"
)

info() { echo -e "\033[1;34m==>\033[0m $*"; }
warn() { echo -e "\033[1;33mWARNING:\033[0m $*"; }
fail() { echo -e "\033[1;31mERROR:\033[0m $*"; exit 1; }

# Asks a yes/no question, returns 0 for yes. $2 is the default answer (y or n).
ask() {
    local prompt answer
    if [[ $2 == y ]]; then prompt="[Y/n]"; else prompt="[y/N]"; fi
    while true; do
        read -r -p "$1 $prompt " answer || fail "No answer (end of input). Run this script from a terminal."
        answer=${answer:-$2}
        case ${answer,,} in
            y|yes) return 0 ;;
            n|no) return 1 ;;
        esac
    done
}

is_online() {
    if command -v wget > /dev/null; then
        wget -q --spider --timeout=10 --tries=1 "$REPO_URL_CHECK"
    else
        curl -s --head --max-time 10 "$REPO_URL_CHECK" > /dev/null
    fi
}

# Name of the Wi-Fi network wlan0 is connected to (empty if none)
wifi_connection() {
    nmcli -t -g GENERAL.CONNECTION device show wlan0 2> /dev/null | grep -v '^--$' || true
}

check_online() {
    info "Checking the internet connection..."
    while ! is_online; do
        local wifi
        wifi=$(wifi_connection)
        if [[ -n $wifi ]]; then
            warn "The Portiloop is connected to the Wi-Fi network \"$wifi\", but cannot reach $REPO_URL_CHECK."
        else
            warn "The Portiloop is not connected to the internet."
        fi
        echo "Please connect it to a Wi-Fi network with internet access."
        echo "(If you are connected through the Portiloop's own Wi-Fi, it may drop for a few seconds.)"
        if ask "Open nmtui to connect to a Wi-Fi network (choose \"Activate a connection\")?" y; then
            sudo nmtui || true
        elif ! ask "Check the connection again?" y; then
            fail "The Portiloop must be online to update. Nothing was changed."
        fi
    done
    info "The Portiloop is online."
}

# Fails if a recording is running: the CSV of a recording stays open until it stops
check_no_recording() {
    local open_csv
    open_csv=$(find /proc/[0-9]*/fd -lname '*.csv' 2> /dev/null | head -n 1)
    if [[ -n $open_csv ]]; then
        fail "A recording seems to be running ($(readlink "$open_csv")). Stop it, then run this script again."
    fi
}

check_branch() {
    local branch
    branch=$(git symbolic-ref --short -q HEAD || true)
    if [[ -z $branch ]]; then
        fail "The software is not on a branch (detached HEAD). Run 'git checkout main' first, or ask for help."
    fi
    if [[ $branch != main ]]; then
        warn "The software is on the branch \"$branch\", not \"main\"."
        ask "Update the branch \"$branch\" anyway?" n || fail "Nothing was changed."
    fi
}

check_local_changes() {
    local changes
    changes=$(git status --porcelain --untracked-files=no)
    [[ -z $changes ]] && return 0
    warn "These files of the Portiloop software were modified on this Portiloop:"
    echo "$changes"
    echo "They must be put aside (stashed) before updating. You can get them back later with 'git stash pop'."
    if ask "Stash these modifications?" n; then
        git stash push -m "update.sh $(date '+%Y-%m-%d %H:%M')" || fail "Could not stash the modifications."
        info "Modifications stashed (see them with 'git stash list')."
    else
        fail "Nothing was changed. Ask for help if you are not sure what these modifications are."
    fi
}

pull() {
    OLD_HEAD=$(git rev-parse HEAD)
    info "Downloading the latest version..."
    if ! git pull --ff-only; then
        fail "Could not update the software (see the message above). Nothing was changed. Please ask for help."
    fi
    NEW_HEAD=$(git rev-parse HEAD)
    if git lfs version > /dev/null 2>&1; then
        git lfs pull || warn "Could not download the large files (sounds, models) with git lfs."
    fi
    if [[ $OLD_HEAD == "$NEW_HEAD" ]]; then
        info "The software was already up to date."
        return
    fi
    info "Updated from $(git rev-parse --short "$OLD_HEAD") to $(git rev-parse --short "$NEW_HEAD"):"
    git log --format='  - %s' "$OLD_HEAD..$NEW_HEAD"
    if ! git diff --quiet "$OLD_HEAD" "$NEW_HEAD" -- setup.py pyproject.toml 'requirements*.txt'; then
        if [[ -x $PIP ]]; then
            info "The dependencies changed, reinstalling the Portiloop package [this may take a while]..."
            "$PIP" install -e . || fail "Could not install the dependencies. Please ask for help."
        else
            warn "The dependencies changed, but $PIP was not found: reinstall the package manually."
        fi
    fi
}

# Jupyter service of this installation: miniforge_jupyter.service (installed by `make`) or jupyter.service
jupyter_service_source() {
    if [[ -f /etc/systemd/system/jupyter.service ]] && ! grep -q miniforge /etc/systemd/system/jupyter.service; then
        echo jupyter.service
    else
        echo miniforge_jupyter.service
    fi
}

install_system_files() {
    local entry src dst to_install=()
    for entry in "$(jupyter_service_source) /etc/systemd/system/jupyter.service" "${SYSTEM_FILES[@]}"; do
        read -r src dst <<< "$entry"
        if ! cmp -s "$SETUP_DIR/$src" "$dst"; then
            to_install+=("$entry")
        fi
    done
    if [[ ${#to_install[@]} -eq 0 ]]; then
        info "The system files are up to date."
        return
    fi
    info "These system files are new or changed:"
    for entry in "${to_install[@]}"; do
        read -r src dst <<< "$entry"
        echo "  - $dst"
    done
    if ! ask "Install them (your password may be asked)?" y; then
        warn "System files not installed: some features of this version may not work."
        return
    fi
    for entry in "${to_install[@]}"; do
        read -r src dst <<< "$entry"
        if [[ -f $dst ]]; then
            sudo cp "$dst" "$dst.bak" || fail "Could not back up $dst."
        fi
        sudo cp "$SETUP_DIR/$src" "$dst" || fail "Could not install $dst."
    done
    sudo systemctl daemon-reload
    sudo udevadm control --reload || true
    SYSTEM_FILES_CHANGED=1
    info "System files installed (previous versions saved as .bak)."
}

offer_reboot() {
    if [[ $OLD_HEAD == "$NEW_HEAD" && -z $SYSTEM_FILES_CHANGED ]]; then
        info "Nothing changed, no reboot needed."
        return
    fi
    echo "The Portiloop must reboot to use the new version."
    if ask "Reboot now?" y; then
        info "Rebooting..."
        sudo reboot
    else
        warn "Reboot the Portiloop before the next recording ('sudo reboot')."
    fi
}

main() {
    [[ $EUID -eq 0 ]] && fail "Run this script as mendel, without sudo: bash ~/portiloop-software/update.sh"
    # The script can also be run from outside the software folder (e.g., downloaded for the first update)
    local script_dir
    script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
    if [[ -d $script_dir/.git && -d $script_dir/$SETUP_DIR ]]; then
        cd "$script_dir" || fail "Could not open $script_dir."
    else
        cd "$HOME/portiloop-software" || fail "Could not find the Portiloop software folder ~/portiloop-software."
    fi
    if [[ $1 == --after-pull ]]; then
        OLD_HEAD=$2
        NEW_HEAD=$3
    else
        info "Updating the Portiloop software in $PWD"
        check_no_recording
        check_branch
        check_online
        check_local_changes
        pull
        exec bash ./update.sh --after-pull "$OLD_HEAD" "$NEW_HEAD"
    fi
    install_system_files
    offer_reboot
}

main "$@"; exit $?
