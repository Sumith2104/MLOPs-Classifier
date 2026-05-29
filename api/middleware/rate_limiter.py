import os
import time
import yaml
from collections import deque
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple

# Load config to get rate limit settings
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "training" / "configs" / "train_config.yaml"

class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = {}

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = deque()
            
        history = self.requests[client_id]
        while history and history[0] < now - self.window_seconds:
            history.popleft()
            
        if len(history) < self.limit:
            history.append(now)
            return True, 0
            
        oldest_req = history[0]
        retry_after = int(max(1, oldest_req + self.window_seconds - now))
        return False, retry_after


class RedisRateLimiter:
    """
    Redis sliding window rate limiter.
    Uses a sorted set (ZSET) to store timestamps.
    """
    def __init__(self, limit: int, window_seconds: int, redis_url: str):
        self.limit = limit
        self.window_seconds = window_seconds
        self.redis_url = redis_url
        self.redis_client = None
        # We lazy-load redis to avoid dependency issues if Redis is not installed
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
        except ImportError:
            print("[RateLimiter] WARNING: redis package not found. Falling back to in-memory limiter.")

    def is_allowed(self, client_id: str, key_prefix: str) -> Tuple[bool, int]:
        if not self.redis_client:
            return True, 0 # Fallback safety
            
        key = f"ratelimit:{key_prefix}:{client_id}"
        now = time.time()
        clear_before = now - self.window_seconds
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, clear_before)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.window_seconds)
            pipe.zrange(key, 0, 0, withscores=True)
            
            _, current_count, _, _, oldest_item = pipe.execute()
            
            if current_count < self.limit:
                return True, 0
                
            self.redis_client.zrem(key, str(now))
            
            oldest_score = oldest_item[0][1] if oldest_item else now
            retry_after = int(max(1, oldest_score + self.window_seconds - now))
            return False, retry_after
        except Exception as e:
            print(f"[RateLimiter] Redis connection/execution error: {e}. Falling back to allow.")
            return True, 0


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        
        # Load configs
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            
        serving_cfg = cfg.get("serving", {})
        rl_cfg = serving_cfg.get("rate_limiting", {})
        
        self.enabled = rl_cfg.get("enabled", True)
        
        # Limits
        self.single_limit = rl_cfg.get("single_window_limit", 60)
        self.single_window = rl_cfg.get("single_window_seconds", 60)
        self.batch_limit = rl_cfg.get("batch_window_limit", 10)
        self.batch_window = rl_cfg.get("batch_window_seconds", 60)
        
        # Redis setup
        self.redis_enabled = rl_cfg.get("redis_enabled", False)
        self.redis_url = rl_cfg.get("redis_url", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        
        if self.enabled:
            if self.redis_enabled:
                print(f"[RateLimiter] Initializing Redis sliding-window limiters")
                self.single_limiter = RedisRateLimiter(self.single_limit, self.single_window, self.redis_url)
                self.batch_limiter = RedisRateLimiter(self.batch_limit, self.batch_window, self.redis_url)
            else:
                print(f"[RateLimiter] Initializing In-Memory sliding-window limiters")
                self.single_limiter = InMemoryRateLimiter(self.single_limit, self.single_window)
                self.batch_limiter = InMemoryRateLimiter(self.batch_limit, self.batch_window)
        else:
            print(f"[RateLimiter] Rate limiting is disabled in configuration")

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)
            
        path = request.url.path
        
        # Rate limit only /predict, /invocations, and /predict/batch endpoints
        is_single = (path == "/predict") or (path == "/invocations")
        is_batch = (path == "/predict/batch")
        
        if not (is_single or is_batch):
            return await call_next(request)
            
        # Get client IP
        client_ip = request.headers.get("x-forwarded-for")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
            
        # Apply rate limit
        allowed = True
        retry_after = 0
        
        if is_single:
            if self.redis_enabled and isinstance(self.single_limiter, RedisRateLimiter):
                allowed, retry_after = self.single_limiter.is_allowed(client_ip, "single")
            else:
                allowed, retry_after = self.single_limiter.is_allowed(client_ip)
        elif is_batch:
            if self.redis_enabled and isinstance(self.batch_limiter, RedisRateLimiter):
                allowed, retry_after = self.batch_limiter.is_allowed(client_ip, "batch")
            else:
                allowed, retry_after = self.batch_limiter.is_allowed(client_ip)
                
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please try again later.",
                    "type": "single" if is_single else "batch",
                    "limit": self.single_limit if is_single else self.batch_limit,
                    "window_seconds": self.single_window if is_single else self.batch_window
                },
                headers={"Retry-After": str(retry_after)}
            )
            
        return await call_next(request)
