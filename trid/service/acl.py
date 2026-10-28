"""Access control: a passage is visible iff its document's tags intersect the user's tags. Deny by default:
an unknown user, a user without tags or a document without tags sees/is seen by nothing.

Document tags, in priority order: `<file>.acl.json` next to the file (`{"acl": ["staff", "hr"]}` or a bare list),
then the nearest `acl.json` in the file's directory or a parent up to the ingest root
(`{"default": ["staff"], "files": {"급여규정.txt": ["hr"]}}`, `files` keyed relative to that directory).
User tags: `users.json` (`{"user": ["tag", ...]}`).
"""
import json
from pathlib import Path


def _tags(value):
    if isinstance(value, dict):
        value = value.get("acl", value.get("tags", []))
    if not isinstance(value, list) or not all(isinstance(t, str) and t for t in value):
        raise ValueError("ACL 태그는 비어 있지 않은 문자열의 목록이어야 합니다.")
    return set(value)


def document_tags(path, root):
    path, root = Path(path), Path(root)
    sidecar = path.with_name(path.name + ".acl.json")
    if sidecar.exists():
        return _tags(json.loads(sidecar.read_text(encoding="utf-8")))
    directory = path.parent
    while True:
        config = directory / "acl.json"
        if config.exists():
            spec = json.loads(config.read_text(encoding="utf-8"))
            files = spec.get("files", {})
            rel = str(path.relative_to(directory)).replace("\\", "/")
            if rel in files:
                return _tags(files[rel])
            if "default" in spec:
                return _tags(spec["default"])
        if directory == root or directory == directory.parent:
            return set()
        directory = directory.parent


def load_users(path):
    """{user: set(tags)}. Entries are either `[tag, ...]` (local CLI only) or
    `{"tags": [...], "token_sha256": "<hex>"}`; only entries with a token can authenticate over HTTP."""
    path = Path(path)
    if not path.exists():
        return {}
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("users.json은 {사용자: [태그, ...] | {tags, token_sha256}} 형식이어야 합니다.")
    users = {}
    for user, entry in spec.items():
        if isinstance(entry, dict):
            users[user] = _TaggedUser(_tags(entry.get("tags", [])), entry.get("token_sha256"))
        else:
            users[user] = _tags(entry)
    return users


class _TaggedUser(set):
    """Tag set that also remembers the sha256 of the user's bearer token (None: CLI only)."""
    def __init__(self, tags, token_sha256):
        super().__init__(tags)
        self.token_sha256 = token_sha256


def user_tags(users, user):
    return set(users.get(user, ()))


def authenticate(users, token):
    """Bearer token -> user name, or None. Constant-time compare against every stored hash."""
    import hashlib
    import hmac
    if not isinstance(token, str) or not token:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    found = None
    for user, tags in users.items():
        stored = getattr(tags, "token_sha256", None)
        if stored and hmac.compare_digest(stored, digest):
            found = user
    return found


def issue_token(path, user):
    """Create a random token for `user` in users.json, store only its sha256, return the token once."""
    import hashlib
    import secrets
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if user not in spec:
        raise ValueError(f"users.json에 사용자 {user!r}가 없습니다. 태그를 먼저 등록하세요.")
    entry = spec[user] if isinstance(spec[user], dict) else {"tags": spec[user]}
    token = secrets.token_urlsafe(32)
    entry["token_sha256"] = hashlib.sha256(token.encode("utf-8")).hexdigest()
    spec[user] = entry
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return token


def visible(doc_tags, tags):
    return bool(set(doc_tags) & set(tags))
