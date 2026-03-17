#!/bin/bash
# Setup rclone for SOTA R2 bucket access (read-only)

set -e

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "rclone not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install rclone
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl https://rclone.org/install.sh | sudo bash
    else
        echo "Please install rclone manually: https://rclone.org/install/"
        exit 1
    fi
fi

echo "Configuring rclone for SOTA R2 bucket..."

# Create rclone config directory
mkdir -p ~/.config/rclone

# Check if sota remote already exists
if rclone listremotes | grep -q "^sota:"; then
    echo "Warning: 'sota' remote already exists in rclone config."
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    # Remove existing config
    rclone config delete sota
fi

# Add the SOTA remote config
# NOTE: Replace <ACCOUNT_ID>, <ACCESS_KEY>, and <SECRET_KEY> with actual values
cat >> ~/.config/rclone/rclone.conf << 'EOF'

[sota]
type = s3
provider = Cloudflare
access_key_id = <ACCESS_KEY>
secret_access_key = <SECRET_KEY>
endpoint = https://b16f4a3487a8546e6a300d1a7494c334.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
EOF

echo ""
echo "rclone configured successfully!"
echo ""
echo "Test with: rclone lsd sota:sota"
echo ""
echo "If this fails, edit ~/.config/rclone/rclone.conf and replace:"
echo "  <ACCESS_KEY> with your access key ID"
echo "  <SECRET_KEY> with your secret access key"
