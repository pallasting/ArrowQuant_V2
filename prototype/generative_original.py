    import os                                                                  
     import json                                                                
     from requests import post # Changed from 'import requests' to 'from        
 requests import post' for clarity                                              
                                                                                
     # --- LLM API Configuration (from the test_api_models.py) ---              
     LLM_API_KEY = "sk-0437c02b1560470981866f50b05759e3"                        
     LLM_BASE_URL = "http://192.168.1.99:8046"                                  
     LLM_ENDPOINT = "/gemini-antigravity/v1/messages" # Using Gemini            
 Antigravity Claude protocol                                                    
     LLM_MODEL = "gemini-claude-opus-4-6-thinking" # The most 'thinking'        
 capable model                                                                  
                                                                                
     def llm_call_for_compression(prompt: str) -> str:                          
         """Calls the configured LLM API to get a response."""                  
         full_url = f"{LLM_BASE_URL}{LLM_ENDPOINT}"                             
         headers = {                                                            
             "Content-Type": "application/json",                                
             "X-API-Key": LLM_API_KEY                                           
         }                                                                      
         payload = {                                                            
             "model": LLM_MODEL,                                                
             "max_tokens": 1000,                                                
             "messages": [{"role": "user", "content": prompt}]                  
         }                                                                      
         try:                                                                   
             response = post(full_url, headers=headers, json=payload,           
 timeout=60)                                                                    
             response.raise_for_status()                                        
             json_response = response.json()                                    
             if json_response.get("content") and                                
 isinstance(json_response["content"], list):                                    
                 # Concatenate all text blocks                                  
                 return " ".join([block.get("text", "") for block in            
 json_response["content"] if block.get("type") == "text"])                      
             return str(json_response) # Fallback for non-text content or other 
 structures 
        except Exception as e:                                                 
             print(f"Error calling LLM for compression: {e}")                   
             return f"Error: {e}"                                               
                                                                                
     def compress_text_generative(text: str) -> dict:                           
         """                                                                    
         Compresses text using an LLM to generate a summary and extract key     
 entities.                                                                      
         """                                                                    
         print(f"Calling LLM for text summary...")                              
         prompt_summary = f"Summarize the following text concisely. Text:       
 {text}"                                                                        
         summary = llm_call_for_compression(prompt_summary)                     
                                                                                
         print(f"Calling LLM for entity extraction...")                         
         # Request entities in JSON list format for easier parsing              
         prompt_entities = f"Extract key entities (e.g., all numbers, names,    
 dates, locations, key terms) from the following text and return as a JSON list 
 of strings. Example: ['entity1', 'entity2']. Text: {text}"                     
         entities_str = llm_call_for_compression(prompt_entities)               
                                                                                
         entities_list = []                                                     
         try:                                                                   
             # Try to parse as JSON list                                        
             parsed_entities = json.loads(entities_str)                         
             if isinstance(parsed_entities, list):                              
                 entities_list = parsed_entities                                
             else:                                                              
                 print(f"Warning: LLM returned non-list JSON for entities:      
 {entities_str[:200]}")                                                         
                 entities_list = [entities_str[:200]] # Store raw output if not 
 a list                                                                         
         except json.JSONDecodeError:                                           
             print(f"Warning: LLM did not return valid JSON for entities:       
 {entities_str[:200]}")                                                         
             entities_list = [entities_str[:200]] # Store raw output if not     
 valid JSON                                                                     
        
         # Calculate compression metrics                                        
         original_len = len(text.encode('utf-8'))                               
         summary_len = len(summary.encode('utf-8'))                             
                                                                                
         return {                                                               
             "summary": summary,                                                
             "entities": entities_list,                                         
             "original_length": original_len,                                   
             "compressed_length_estimate": summary_len,                         
             "llm_model": LLM_MODEL,                                            
             "status": "LLM Compressed"                                         
         }                                                                      
                                                                                
     # Original mock/simple logic - Keeping a placeholder or example here       
     # This part depends on the original content of your generative.py          
     # If there was a simpler `compress_text_generative` before, this           
 LLM-integrated one replaces it.                                                
     def mock_compress_text_generative(text: str) -> dict:                      
         """A simple, non-LLM based compression for comparison."""              
         summary = text[:min(len(text), 50)] + "..."                            
         entities = ["mock_entity_1", "mock_entity_2"]                          
         original_len = len(text.encode('utf-8'))                               
         compressed_len = len(summary.encode('utf-8'))                          
         return {                                                               
             "summary": summary,                                                
             "entities": entities,                                              
             "original_length": original_len,                                   
             "compressed_length_estimate": compressed_len,                      
             "llm_model": "Mock",                                               
             "status": "Mock Compressed"                                        
         }                                                                      
                                                                                
     # You might want to choose between `compress_text_generative` and          
 `mock_compress_text_generative`                                                
     # The benchmark will call `compress_text_generative` by default.    