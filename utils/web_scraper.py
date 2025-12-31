"""
Web Scraper Module
Scrapes content from websites and prepares it for training
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
from typing import List, Dict, Set
import re


class WebScraper:
    """
    Scrapes websites and extracts content for chatbot training
    """
    
    def __init__(self):
        self.visited_urls = set()
        self.scraped_data = []
        self.all_scraped_data = []  # Store all data across multiple sites
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_website(self, url: str, max_pages: int = 50, same_domain_only: bool = True) -> List[Dict]:
        """
        Scrape a website and extract content
        
        Args:
            url: Starting URL
            max_pages: Maximum number of pages to scrape
            same_domain_only: Only scrape pages from the same domain
            
        Returns:
            List of scraped page data
        """
        print(f"🔍 Starting to scrape: {url}")
        
        base_domain = urlparse(url).netloc
        urls_to_visit = [url]
        
        while urls_to_visit and len(self.visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            try:
                page_data = self._scrape_page(current_url)
                
                if page_data:
                    self.scraped_data.append(page_data)
                    self.visited_urls.add(current_url)
                    
                    print(f"✓ Scraped ({len(self.visited_urls)}/{max_pages}): {current_url}")
                    
                    # Find more links to visit
                    if same_domain_only:
                        new_links = [
                            link for link in page_data.get('links', [])
                            if urlparse(link).netloc == base_domain
                            and link not in self.visited_urls
                        ]
                        urls_to_visit.extend(new_links[:5])  # Limit to 5 new links per page
                
                # Be polite - delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Error scraping {current_url}: {e}")
                continue
        
        print(f"\n✅ Scraping complete! Total pages: {len(self.scraped_data)}")
        return self.scraped_data
    
    def _scrape_page(self, url: str) -> Dict:
        """
        Scrape a single page and extract content
        
        Args:
            url: Page URL
            
        Returns:
            Dictionary with page data
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ''
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Extract meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc.get('content', '').strip() if meta_desc else ''
            
            # Extract links
            links = self._extract_links(soup, url)
            
            # Extract headings
            headings = self._extract_headings(soup)
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'content': content,
                'headings': headings,
                'links': links,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error in _scrape_page for {url}: {e}")
            return None
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit content length
        return text[:5000]  # First 5000 characters
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only include http/https links
            if full_url.startswith(('http://', 'https://')):
                links.append(full_url)
        
        return list(set(links))[:20]  # Return unique links, max 20
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings from page"""
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = heading.get_text().strip()
            if text:
                headings.append(text)
        return headings
    
    def scrape_multiple_websites(self, urls: List[str], max_pages_per_site: int = 20) -> List[Dict]:
        """
        Scrape multiple websites
        
        Args:
            urls: List of starting URLs
            max_pages_per_site: Max pages per site
            
        Returns:
            Combined list of all scraped data
        """
        self.all_scraped_data = []  # Reset for fresh scraping
        
        for url in urls:
            print(f"\n{'='*60}")
            print(f"Scraping site: {url}")
            print(f"{'='*60}\n")
            
            # Reset for each site
            self.visited_urls = set()
            self.scraped_data = []
            
            site_data = self.scrape_website(url, max_pages=max_pages_per_site)
            self.all_scraped_data.extend(site_data)  # Add to combined data
            
            print(f"\n✓ Completed {url}: {len(site_data)} pages\n")
        
        return self.all_scraped_data
    
    def save_to_file(self, filename: str = 'website_scraped_data.json'):
        """Save all scraped data to JSON file"""
        filepath = f'data/{filename}'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.all_scraped_data)} pages to {filepath}")
    
    def convert_to_training_format(self) -> List[Dict]:
        """
        Convert scraped data to chatbot training format
        
        Returns:
            List of training conversations
        """
        training_data = []
        
        for page in self.all_scraped_data:
            # Create Q&A pairs from the content
            
            # 1. Title-based question
            if page['title']:
                training_data.append({
                    'user': f"Tell me about {page['title']}",
                    'bot': f"{page['description'] or page['content'][:500]}... Learn more at {page['url']}",
                    'metadata': {
                        'source': 'website',
                        'url': page['url']
                    }
                })
            
            # 2. Heading-based questions
            for heading in page['headings'][:3]:  # Top 3 headings
                training_data.append({
                    'user': heading,
                    'bot': f"{page['content'][:400]}... For more details, visit {page['url']}",
                    'metadata': {
                        'source': 'website',
                        'url': page['url']
                    }
                })
            
            # 3. URL reference question
            domain = urlparse(page['url']).netloc
            training_data.append({
                'user': f"What information do you have from {domain}?",
                'bot': f"I have information about {page['title']}. {page['description']}. Check it out here: {page['url']}",
                'metadata': {
                    'source': 'website',
                    'url': page['url']
                }
            })
        
        print(f"📚 Generated {len(training_data)} training examples from {len(self.all_scraped_data)} pages")
        return training_data


def scrape_and_train(urls: List[str], max_pages_per_site: int = 20):
    """
    Main function to scrape websites and prepare training data
    
    Args:
        urls: List of URLs to scrape
        max_pages_per_site: Maximum pages per site
    """
    scraper = WebScraper()
    
    # Scrape websites
    all_data = scraper.scrape_multiple_websites(urls, max_pages_per_site)
    
    # Save raw data
    scraper.save_to_file('website_scraped_data.json')
    
    # Convert to training format
    training_data = scraper.convert_to_training_format()
    
    # Save training data
    with open('data/website_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ All done! Ready to train chatbot with {len(training_data)} examples")
    
    return training_data


if __name__ == '__main__':
    # Example usage
    websites = [
        'https://example.com',
        'https://yourblog.com',
        'https://yourportfolio.com'
    ]
    
    scrape_and_train(websites, max_pages_per_site=15)
