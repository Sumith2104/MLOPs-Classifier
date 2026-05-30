import asyncio
import logging
from datetime import datetime, timezone
from api.database import get_supabase
from api.routes.predict import get_predictor, get_flag_manager
from monitoring.logger import get_logger

logger = get_logger(__name__)

async def start_query_classifier_worker():
    """Background task to poll the Supabase 'client_queries' table for unprocessed queries,
    classify them in batches, and save them in 'classified_queries'."""
    logger.info("Supabase queries background classifier worker started.")
    
    while True:
        try:
            db = get_supabase()
            if db is None:
                # Supabase not configured, sleep and retry
                await asyncio.sleep(10)
                continue
                
            # 1. Fetch unprocessed queries (limit to 50 max batch size)
            res = db.table("client_queries").select("id, message, client_app_id").eq("processed", False).limit(50).execute()
            rows = res.data if res else []
            
            if not rows:
                await asyncio.sleep(5)
                continue
                
            logger.info(f"Background worker found {len(rows)} unclassified queries.")
            
            row_ids = [r["id"] for r in rows]
            messages = [r["message"] for r in rows]
            client_app_ids = [r.get("client_app_id", "webapp") for r in rows]
            
            # 2. Mark them as processed immediately to prevent duplicate classification
            db.table("client_queries").update({"processed": True}).in_("id", row_ids).execute()
            
            # 3. Classify queries
            predictor = get_predictor()
            flag_manager = get_flag_manager()
            
            # Run model batch prediction in separate thread to prevent blocking FastAPI event loop
            results = await asyncio.to_thread(predictor.predict_batch, messages)
            
            # 4. Insert results into classified_queries
            classified_records = []
            
            for row_id, message, client_app_id, res_item in zip(row_ids, messages, client_app_ids, results):
                is_error = "error" in res_item and res_item["error"] is not None
                
                flagged = True
                if not is_error:
                    flagged = flag_manager.is_flagged(
                        res_item.get("intent_confidence", 1.0),
                        res_item.get("priority_confidence", 1.0)
                    )
                    
                record = {
                    "query_id": row_id,
                    "message": message,
                    "intent": res_item.get("intent") if not is_error else None,
                    "priority": res_item.get("priority") if not is_error else None,
                    "intent_confidence": res_item.get("intent_confidence") if not is_error else None,
                    "priority_confidence": res_item.get("priority_confidence") if not is_error else None,
                    "flagged": flagged,
                    "error": res_item.get("error") if is_error else None,
                    "client_app_id": client_app_id
                }
                classified_records.append(record)
                
            if classified_records:
                db.table("classified_queries").insert(classified_records).execute()
                logger.info(f"Successfully classified and stored {len(classified_records)} queries in classified_queries.")
                
        except asyncio.CancelledError:
            logger.info("Queries background classifier worker cancelled. Shutting down.")
            break
        except Exception as e:
            logger.error(f"Error in background query classifier worker: {e}", exc_info=True)
            
        await asyncio.sleep(5)
