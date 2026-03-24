#!/bin/bash

# AutoSecTester - Automated Web Application Security Tester
# Usage: ./autosectester.sh https://your-target.com

TARGET="$1"
OUTPUT_DIR="scan-results-$(date +%Y%m%d-%H%M%S)"
API_KEY=""  # Add your VirusTotal/API keys here for enhanced scanning

if [ -z "$TARGET" ]; then
    echo "Usage: $0 https://target-url.com"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "=== Auto Security Tester ==="
echo "Target: $TARGET"
echo "Output: $OUTPUT_DIR"
echo ""

# 1. Nmap Scan - Port Discovery
echo "[1/8] Running Nmap port scan..."
nmap -sV -sC -oA "$OUTPUT_DIR/nmap-scan" "$TARGET" 2>/dev/null

# 2. OWASP ZAP Baseline Scan
echo "[2/8] Running OWASP ZAP baseline scan..."
docker run -v "$(pwd):/zap/wrk:rw" \
    owasp/zap2docker-stable zap-baseline.py \
    -t "$TARGET" -J zap-report.json 2>/dev/null
mv zap-report.json "$OUTPUT_DIR/" 2>/dev/null

# 3. Nikto Web Server Scan
echo "[3/8] Running Nikto scan..."
nikto -h "$TARGET" -o "$OUTPUT_DIR/nikto-report.txt" 2>/dev/null

# 4. SQLMap - SQL Injection Detection
echo "[4/8] Running SQLMap injection tests..."
sqlmap -u "$TARGET" --batch --random-agent \
    --output-dir="$OUTPUT_DIR/sqlmap" 2>/dev/null

# 5. Directory Discovery with dirb
echo "[5/8] Discovering directories..."
dirb "$TARGET" -o "$OUTPUT_DIR/dirb-results.txt" 2>/dev/null

# 6. Nuclei Vulnerability Scan
echo "[6/8] Running nuclei vulnerability scan..."
nuclei -u "$TARGET" -o "$OUTPUT_DIR/nuclei-results.txt" \
    -severity critical,high,medium 2>/dev/null

# 7. XSStrike - XSS Testing
echo "[7/8] Testing for XSS vulnerabilities..."
cd XSStrike 2>/dev/null && python3 xsstrike.py \
    -u "$TARGET" --seed 2>/dev/null && cd ..
mv XSStrike/results.json "$OUTPUT_DIR/xss-results.json" 2>/dev/null

# 8. wfuzz - Fuzzing Parameters
echo "[8/8] Fuzzing parameters..."
wfuzz -c -z file,/usr/share/wfuzz/wordlist/general/common.txt \
    "$TARGET/FUZZ" --hc 404 -o json \
    -w "$OUTPUT_DIR/wfuzz-results.json" 2>/dev/null

# Generate Summary Report
echo ""
echo "=== Scan Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key findings summary:"
[ -f "$OUTPUT_DIR/nmap-scan.gnmap" ] && grep -E "Ports:|open" "$OUTPUT_DIR/nmap-scan.gnmap" | head -5
[ -f "$OUTPUT_DIR/nikto-report.txt" ] && grep -E "^\+ " "$OUTPUT_DIR/nikto-report.txt" | head -10
[ -f "$OUTPUT_DIR/nuclei-results.txt" ] && cat "$OUTPUT_DIR/nuclei-results.txt" | head -10

echo ""
echo "Next steps:"
echo "1. Review detailed reports in $OUTPUT_DIR/"
echo "2. Manually verify critical/high findings"
echo "3. Use Burp Suite for deeper manual testing"
