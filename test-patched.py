import requests
import json
import pandas as pd
import time
from datetime import datetime
import os
import re
import psycopg2
from psycopg2 import sql

def get_page_posts(page_id, access_token, post_limit=300):

    # verify page and token
    verify_url = f"https://graph.facebook.com/v24.0/214973071707113"
    verify_params = {'access_token': 'EAAPC0QetLgoBQXE9TGg5cERxcAocZBQCoeoaEbcJNUdCutII5L7vgYrQiZAZA9Qx6aEE55wZCYnlsrEVMGgKPSjMLcL62r2hKHi0L21OgN2AdBHNnUkbtkQgk0ZBJdEilw8TvtM9O5CQnc5wTGWD3KcBknOpIqN3jPOjHZCRLUZCVZBabW4MPv3jzXwLmHyzIZCJRB7sZD', 'fields': 'name,id'}
    
    verify_response = requests.get(verify_url, params=verify_params)
    verify_data = verify_response.json()
    
    if 'error' in verify_data:
        print(f"Error accessing page: {verify_data['error']['message']}")
        print(f"Error code: {verify_data['error'].get('code')}")
        print(f"Error subcode: {verify_data['error'].get('error_subcode')}")
        print("Detailed error info:", json.dumps(verify_data['error'], indent=2))
        
        # Check common error codes
        error_code = verify_data['error'].get('code')
        if error_code == 803 or error_code == 200:
            print("\nSuggestion: Your access token might be invalid or expired.")
        elif error_code == 100:
            print("\nSuggestion: The page ID might be incorrect or the page doesn't exist.")
        elif error_code == 10:
            print("\nSuggestion: The application may not have the required permissions.")
            
        return []
    
    print(f"Successfully verified page: {verify_data.get('name')} (ID: {verify_data.get('id')})")
    
    # Continue with fetching posts
    base_url = f"https://graph.facebook.com/v24.0/{page_id}/posts"
    
    params = {
        'access_token': access_token,
        'limit': 25,  # Facebook generally limits to 25 per request
        'fields': 'id,message,created_time,permalink_url'
    }
    
    posts_data = []
    next_url = base_url
    
    while next_url and len(posts_data) < post_limit:
        response = requests.get(next_url, params=params)
        data = response.json()
        
        if 'error' in data:
            print(f"Error fetching posts: {data['error']['message']}")
            print(f"Error code: {data['error'].get('code')}")
            break            
        # Extract posts from the response
        if 'data' in data:
            posts_data.extend(data['data'])
        
        # Check if there are more posts to fetch
        if 'paging' in data and 'next' in data['paging'] and len(posts_data) < post_limit:
            next_url = data['paging']['next']
            params = {}  # Parameters are included in the next URL
            time.sleep(1)  # Avoid rate limiting
        else:
            break
    
    return posts_data
def get_comments_for_post(post_id, access_token, comment_limit=500):

    base_url = f"https://graph.facebook.com/v24.0/{post_id}/comments"
    
    params = {
        'access_token': access_token,
        'limit': 25,  # Facebook generally limits to 25 per request
        'fields': 'id,message,created_time,like_count,comment_count,from'
    }
    
    comments_data = []
    next_url = base_url
    
    while next_url and len(comments_data) < comment_limit:
        response = requests.get(next_url, params=params)
        data = response.json()
        
        if 'error' in data:
            print(f"Error fetching comments: {data['error']['message']}")
            break
            
        # Extract comments from the response
        if 'data' in data:
            for comment in data['data']:
                comment_info = {
                    'comment_id': comment.get('id'),
                    'post_id': post_id,
                    'message': comment.get('message'),
                    'created_time': comment.get('created_time'),
                    'like_count': comment.get('like_count', 0),
                    'comment_count': comment.get('comment_count', 0),
                    'user_id': comment.get('from', {}).get('id'),
                    'user_name': comment.get('from', {}).get('name') if comment.get('from') else 'Anonymous'
                }
                comments_data.append(comment_info)
                
                # Get replies to comments if they exist
                if comment.get('comment_count', 0) > 0:
                    replies = get_comment_replies(comment.get('id'), access_token)
                    for reply in replies:
                        reply['parent_comment_id'] = comment.get('id')
                        reply['post_id'] = post_id
                    comments_data.extend(replies)
        
        # Check if there are more comments to fetch
        if 'paging' in data and 'next' in data['paging'] and len(comments_data) < comment_limit:
            next_url = data['paging']['next']
            params = {}  
            time.sleep(1)  # Avoid rate limiting
        else:
            break
    
    return comments_data

def get_comment_replies(comment_id, access_token, reply_limit=100):
        
    base_url = f"https://graph.facebook.com/v24.0/{comment_id}/comments"
    
    params = {
        'access_token': access_token,
        'limit': 25,
        'fields': 'id,message,created_time,like_count,from'
    }
    
    replies_data = []
    next_url = base_url
    
    while next_url and len(replies_data) < reply_limit:
        response = requests.get(next_url, params=params)
        data = response.json()
        
        if 'error' in data:
            print(f"Error fetching replies: {data['error']['message']}")
            break
            
        if 'data' in data:
            for reply in data['data']:
                reply_info = {
                    'comment_id': reply.get('id'),
                    'is_reply': True,
                    'message': reply.get('message'),
                    'created_time': reply.get('created_time'),
                    'like_count': reply.get('like_count', 0),
                    'user_id': reply.get('from', {}).get('id'),
            'user_name': reply.get('from', {}).get('name') if reply.get('from') else 'Anonymous'
                }
                replies_data.append(reply_info)
        
        # Check if there are more replies to fetch
        if 'paging' in data and 'next' in data['paging'] and len(replies_data) < reply_limit:
            next_url = data['paging']['next']
            params = {}
            time.sleep(1)
        else:
            break   
    return replies_data

def scrape_all_page_comments(page_id, access_token, post_limit=300, comment_limit=500):

    print(f"Fetching posts from page {page_id}...")
    posts = get_page_posts(page_id, access_token, post_limit)
    print(f"Found {len(posts)} posts")
    
    all_comments = []
    posts_with_data = []
    
    for i, post in enumerate(posts):
        post_id = post.get('id')
        print(f"Processing post {i+1}/{len(posts)}: {post_id}")
        
        # Add post data
        post_data = {
            'post_id': post_id,
            'message': post.get('message', ''),
            'created_time': post.get('created_time'),
            'permalink_url': post.get('permalink_url', '')
        }
        posts_with_data.append(post_data)
        
        # Get comments for this post
        try:
            comments = get_comments_for_post(post_id, access_token, comment_limit)
            print(f"  Found {len(comments)} comments/replies")
            all_comments.extend(comments)
            
            # Avoid hitting rate limits
            if i < len(posts) - 1:  # Don't sleep after the last post
                time.sleep(2)
                
        except Exception as e:
            print(f"  Error processing post {post_id}: {str(e)}")
    
    # Convert to DataFrames
    posts_df = pd.DataFrame(posts_with_data)
    comments_df = pd.DataFrame(all_comments)
    
    # Convert timestamps to datetime objects
    for df in [posts_df, comments_df]:
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])
    
    return posts_df, comments_df

def save_data_to_csv(posts_df, comments_df, output_dir=None):

    if output_dir is None:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    posts_filename = os.path.join(output_dir, f"medina_posts_{timestamp}.csv")
    comments_filename = os.path.join(output_dir, f"medina_comments_{timestamp}.csv")
    
    posts_df.to_csv(posts_filename, index=False, encoding='utf-8')
    comments_df.to_csv(comments_filename, index=False, encoding='utf-8')
    
    return posts_filename, comments_filename

# Database connection function
def get_db_connection():
    return psycopg2.connect(
        dbname='page_comments',
        user='scrapper_user',
        password='scRaPPer_user',
        host='10.20.10.42',
        port='5432'
    )

def initialize_database():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medina_posts (
                    post_id TEXT PRIMARY KEY,
                    post_message TEXT,
                    post_time TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medina_comments (
                    id SERIAL PRIMARY KEY,
                    comment_id TEXT UNIQUE,  -- Add this field for Facebook's comment ID
                    post_id TEXT REFERENCES medina_posts(post_id),
                    commenter_name TEXT,
                    comment_time TIMESTAMP,
                    comment_message TEXT
                )
            """)
            conn.commit()
    finally:
        conn.close()

def insert_post_to_db(post_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO medina_posts (post_id, post_message, post_time)
                VALUES (%s, %s, %s)
                ON CONFLICT (post_id) DO NOTHING
            """, (
                post_data['post_id'],
                post_data['message'],
                post_data['created_time']
            ))
            conn.commit()
    except Exception as e:
        print(f"Error inserting post: {e}")
    finally:
        conn.close()

def insert_comment_to_db(comment_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO medina_comments 
                (comment_id, post_id, commenter_name, comment_time, comment_message)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (comment_id) DO NOTHING
                RETURNING id
            """, (
                comment_data['comment_id'],  
                comment_data['post_id'],
                comment_data['user_name'],
                comment_data['created_time'],
                comment_data['message']
            ))
            inserted = cursor.fetchone()
            conn.commit()
            return inserted[0] if inserted else None
    except Exception as e:
        print(f"Error inserting comment: {e}")
        return None
    finally:
        conn.close()

def scrape_all_page_comments(page_id, access_token, post_limit=300, comment_limit=500):
    print(f"Fetching posts from page {page_id}...")
    posts = get_page_posts(page_id, access_token, post_limit)
    print(f"Found {len(posts)} posts")
    
    all_comments = []   
    for i, post in enumerate(posts):
        post_id = post.get('id')
        print(f"Processing post {i+1}/{len(posts)}: {post_id}")
        
        post_data = {
            'post_id': post_id,
            'message': post.get('message', ''),
            'created_time': post.get('created_time')
        }
        
        insert_post_to_db(post_data)

        try:
            comments = get_comments_for_post(post_id, access_token, comment_limit)
            print(f"  Found {len(comments)} comments/replies")
            
            for comment in comments:
                # Insert each comment to database
                insert_comment_to_db(comment)
                all_comments.append(comment)
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            print(f"  Error processing post {post_id}: {str(e)}")
    
    posts_df = pd.DataFrame([{
        'post_id': p.get('id'),
        'message': p.get('message', ''),
        'created_time': p.get('created_time'),
        'permalink_url': p.get('permalink_url', '')
    } for p in posts])    
    comments_df = pd.DataFrame(all_comments)

    for df in [posts_df, comments_df]:
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])    
    return posts_df, comments_df

def get_comments_with_post_messages():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.*, p.post_message
                FROM medina_comments c
                JOIN medina_posts p ON c.post_id = p.post_id
            """)
            return cursor.fetchall()
    finally:
        conn.close()

if __name__ == "__main__":

    initialize_database()
    
    PAGE_ID = '214973071707113'
    ACCESS_TOKEN = 'EAAPC0QetLgoBPjBIkipoyLbAHOLfgNpOIdAbn4dPDyCKpyE7g4QK1XKIdVJZBo0w1PsT3c2T2xVgXFXxFWjTYmgdZBUoZAZASDxZCGTNkZBGdoMuiNMSZBZCImG1XoHnYTYKqwrywf8B66hNpP7ECKvZAiJK2bdPUZASnNJlsxLdRK98osCbfZCiUhZC6Hl1GZAT3ZAb8FY6UZD'
    POST_LIMIT = 300
    COMMENT_LIMIT = 500
    OUTPUT_DIR = "facebook_data"  
    try:
        posts_df, comments_df = scrape_all_page_comments(
            PAGE_ID, 
            ACCESS_TOKEN,
            POST_LIMIT,
            COMMENT_LIMIT
        )
        posts_file, comments_file = save_data_to_csv(posts_df, comments_df, OUTPUT_DIR)        
        comments_with_posts = get_comments_with_post_messages()
        print(f"\nFetched {len(comments_with_posts)} comments with post messages")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")