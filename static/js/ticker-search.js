/**
 * Global Ticker Search
 * Provides comprehensive stock ticker search functionality
 * with support for international exchanges including NSE and BSE
 */

class TickerSearchService {
    constructor() {
      this.initialize();
    }
    
    initialize() {
      // Get DOM elements
      this.searchInput = document.getElementById('ticker');
      this.searchButton = document.getElementById('tickerSearchBtn');
      this.suggestionsContainer = document.getElementById('tickerSuggestions');
      
      if (!this.searchInput || !this.suggestionsContainer) return;
      
      // Initialize market data
      this.initializeMarketData();
      
      // Set up event listeners
      this.setupEventListeners();
    }
    
    initializeMarketData() {
      // Market data by regions
      this.marketData = {
        'US': {
          name: 'US Markets',
          description: 'NYSE and NASDAQ',
          suffix: '',
          tickers: [
            { symbol: 'AAPL', name: 'Apple Inc.', popular: true },
            { symbol: 'MSFT', name: 'Microsoft Corporation', popular: true },
            { symbol: 'GOOGL', name: 'Alphabet Inc. (Google)', popular: true },
            { symbol: 'AMZN', name: 'Amazon.com Inc.', popular: true },
            { symbol: 'META', name: 'Meta Platforms Inc.', popular: true },
            { symbol: 'TSLA', name: 'Tesla Inc.', popular: true },
            { symbol: 'NVDA', name: 'NVIDIA Corporation', popular: true },
            { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
            { symbol: 'NFLX', name: 'Netflix Inc.' },
            { symbol: 'DIS', name: 'The Walt Disney Company' },
            { symbol: 'PFE', name: 'Pfizer Inc.' },
            { symbol: 'KO', name: 'The Coca-Cola Company' },
            { symbol: 'MCD', name: 'McDonald\'s Corporation' },
            { symbol: 'WMT', name: 'Walmart Inc.' },
            { symbol: 'V', name: 'Visa Inc.' }
          ]
        },
        'India': {
          name: 'Indian Markets',
          description: 'NSE and BSE',
          suffix: { NSE: '.NS', BSE: '.BO' },
          tickers: [
            { symbol: 'RELIANCE.NS', name: 'Reliance Industries Ltd.', popular: true },
            { symbol: 'TCS.NS', name: 'Tata Consultancy Services Ltd.', popular: true },
            { symbol: 'INFY.NS', name: 'Infosys Ltd.', popular: true },
            { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Ltd.', popular: true },
            { symbol: 'BHARTIARTL.NS', name: 'Bharti Airtel Ltd.', popular: true },
            { symbol: 'ICICIBANK.NS', name: 'ICICI Bank Ltd.', popular: true },
            { symbol: 'HINDUNILVR.NS', name: 'Hindustan Unilever Ltd.' },
            { symbol: 'SBIN.NS', name: 'State Bank of India' },
            { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd.' },
            { symbol: 'WIPRO.NS', name: 'Wipro Ltd.' },
            { symbol: 'AXISBANK.NS', name: 'Axis Bank Ltd.' },
            { symbol: 'MARUTI.NS', name: 'Maruti Suzuki India Ltd.' },
            { symbol: 'HCLTECH.NS', name: 'HCL Technologies Ltd.' },
            { symbol: 'TATASTEEL.NS', name: 'Tata Steel Ltd.' },
            { symbol: 'ITC.NS', name: 'ITC Ltd.' }
          ],
          exchangeMap: {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'SBIN': 'SBIN.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'WIPRO': 'WIPRO.NS'
          }
        },
        'Europe': {
          name: 'European Markets',
          description: 'LSE, XETRA, Euronext',
          suffix: { LSE: '.L', XETRA: '.DE', Euronext: '.PA' },
          tickers: [
            { symbol: 'BP.L', name: 'BP PLC (London)', popular: true },
            { symbol: 'VOD.L', name: 'Vodafone Group PLC (London)' },
            { symbol: 'HSBA.L', name: 'HSBC Holdings PLC (London)' },
            { symbol: 'SAP.DE', name: 'SAP SE (XETRA)', popular: true },
            { symbol: 'SIE.DE', name: 'Siemens AG (XETRA)' },
            { symbol: 'BAS.DE', name: 'BASF SE (XETRA)' },
            { symbol: 'MC.PA', name: 'LVMH Moët Hennessy (Euronext Paris)', popular: true },
            { symbol: 'BNP.PA', name: 'BNP Paribas (Euronext Paris)' },
            { symbol: 'ABI.BR', name: 'Anheuser-Busch InBev (Euronext Brussels)' }
          ]
        },
        'Asia': {
          name: 'Asian Markets',
          description: 'Tokyo, Hong Kong, Shanghai',
          suffix: { Tokyo: '.T', HongKong: '.HK', Shanghai: '.SS', Korea: '.KS' },
          tickers: [
            { symbol: '7203.T', name: 'Toyota Motor Corporation (Tokyo)', popular: true },
            { symbol: '9984.T', name: 'SoftBank Group Corp. (Tokyo)', popular: true },
            { symbol: '9988.HK', name: 'Alibaba Group Holding Ltd. (Hong Kong)', popular: true },
            { symbol: '0700.HK', name: 'Tencent Holdings Ltd. (Hong Kong)', popular: true },
            { symbol: '3690.HK', name: 'Meituan (Hong Kong)' },
            { symbol: '601398.SS', name: 'Industrial and Commercial Bank of China (Shanghai)' },
            { symbol: '601988.SS', name: 'Bank of China (Shanghai)' },
            { symbol: '005930.KS', name: 'Samsung Electronics Co., Ltd. (Korea)', popular: true },
            { symbol: '000660.KS', name: 'SK Hynix Inc. (Korea)' }
          ]
        },
        'Indices': {
          name: 'Global Indices',
          description: 'Major market indices',
          tickers: [
            { symbol: '^GSPC', name: 'S&P 500 (US)', popular: true },
            { symbol: '^DJI', name: 'Dow Jones Industrial Average (US)', popular: true },
            { symbol: '^IXIC', name: 'NASDAQ Composite (US)', popular: true },
            { symbol: '^NSEI', name: 'NIFTY 50 (India)', popular: true },
            { symbol: '^BSESN', name: 'S&P BSE SENSEX (India)', popular: true },
            { symbol: '^FTSE', name: 'FTSE 100 (UK)' },
            { symbol: '^GDAXI', name: 'DAX (Germany)' },
            { symbol: '^FCHI', name: 'CAC 40 (France)' },
            { symbol: '^N225', name: 'Nikkei 225 (Japan)' },
            { symbol: '^HSI', name: 'Hang Seng Index (Hong Kong)' }
          ]
        },
        'Crypto': {
          name: 'Cryptocurrencies',
          description: 'Digital assets',
          suffix: '-USD',
          tickers: [
            { symbol: 'BTC-USD', name: 'Bitcoin', popular: true },
            { symbol: 'ETH-USD', name: 'Ethereum', popular: true },
            { symbol: 'SOL-USD', name: 'Solana' },
            { symbol: 'ADA-USD', name: 'Cardano' },
            { symbol: 'XRP-USD', name: 'XRP' },
            { symbol: 'DOGE-USD', name: 'Dogecoin' },
            { symbol: 'DOT-USD', name: 'Polkadot' },
            { symbol: 'AVAX-USD', name: 'Avalanche' }
          ]
        }
      };
      
      // Create a flat list of all tickers for quick search
      this.allTickers = [];
      Object.keys(this.marketData).forEach(region => {
        this.marketData[region].tickers.forEach(ticker => {
          this.allTickers.push({
            symbol: ticker.symbol,
            name: ticker.name,
            region: region,
            popular: ticker.popular || false
          });
        });
      });
    }
    
    setupEventListeners() {
      // Search button click
      if (this.searchButton) {
        this.searchButton.addEventListener('click', () => {
          this.handleSearch();
        });
      }
      
      // Input keyup event (search as you type)
      this.searchInput.addEventListener('input', () => {
        this.handleSearch();
      });
      
      // Input focus event
      this.searchInput.addEventListener('focus', () => {
        if (this.searchInput.value.length > 0) {
          this.handleSearch();
        }
      });
      
      // Handle click on suggestion
      this.suggestionsContainer.addEventListener('click', (e) => {
        const suggestion = e.target.closest('.suggestion-item');
        if (suggestion) {
          this.selectSuggestion(suggestion.dataset.symbol);
        }
      });
      
      // Close suggestions when clicking outside
      document.addEventListener('click', (e) => {
        if (e.target !== this.searchInput && e.target !== this.searchButton) {
          this.hideSuggestions();
        }
      });
      
      // Handle special case for Indian stocks (adding .NS suffix)
      this.searchInput.addEventListener('blur', () => {
        setTimeout(() => {
          this.handleIndianStocks();
        }, 200);
      });
    }
    
    handleSearch() {
      const query = this.searchInput.value.trim();
      
      if (query.length < 1) {
        this.hideSuggestions();
        return;
      }
      
      // First check for exact match
      const exactMatch = this.allTickers.find(ticker => 
        ticker.symbol.toLowerCase() === query.toLowerCase()
      );
      
      if (exactMatch) {
        this.selectSuggestion(exactMatch.symbol);
        return;
      }
      
      // Show suggestions
      this.showSuggestions(query);
    }
    
    showSuggestions(query) {
      query = query.toLowerCase();
      
      // Gather matches by region
      const regionMatches = {};
      let totalMatches = 0;
      
      // Check each region for matches
      Object.keys(this.marketData).forEach(region => {
        const matches = this.marketData[region].tickers.filter(ticker => 
          ticker.symbol.toLowerCase().includes(query) || 
          ticker.name.toLowerCase().includes(query)
        );
        
        if (matches.length > 0) {
          regionMatches[region] = matches;
          totalMatches += matches.length;
        }
      });
      
      // Generate HTML for suggestions
      let html = '';
      
      if (totalMatches > 0) {
        // For each region with matches
        Object.keys(regionMatches).forEach(region => {
          const matches = regionMatches[region];
          const marketInfo = this.marketData[region];
          
          html += `
            <div class="suggestion-group">
              <div class="suggestion-market">${marketInfo.name} (${marketInfo.description || ''})</div>
              <div class="suggestion-items">
          `;
          
          // Sort by popularity first, then by whether query is in the ticker symbol
          matches.sort((a, b) => {
            // Popular items first
            if (a.popular && !b.popular) return -1;
            if (!a.popular && b.popular) return 1;
            
            // Exact matches in symbol
            if (a.symbol.toLowerCase() === query) return -1;
            if (b.symbol.toLowerCase() === query) return 1;
            
            // Starts with query in symbol
            if (a.symbol.toLowerCase().startsWith(query) && !b.symbol.toLowerCase().startsWith(query)) return -1;
            if (!a.symbol.toLowerCase().startsWith(query) && b.symbol.toLowerCase().startsWith(query)) return 1;
            
            // Contains query in symbol
            if (a.symbol.toLowerCase().includes(query) && !b.symbol.toLowerCase().includes(query)) return -1;
            if (!a.symbol.toLowerCase().includes(query) && b.symbol.toLowerCase().includes(query)) return 1;
            
            // Alphabetical order
            return a.symbol.localeCompare(b.symbol);
          });
          
          // Limit to first 5 matches per region
          const displayedMatches = matches.slice(0, 5);
          
          displayedMatches.forEach(ticker => {
            html += `
              <div class="suggestion-item" data-symbol="${ticker.symbol}">
                <span class="suggestion-symbol">${ticker.symbol}</span>
                <span class="suggestion-name">${ticker.name}</span>
              </div>
            `;
          });
          
          // Show "more results" if there are more than 5 matches
          if (matches.length > 5) {
            html += `
              <div class="suggestion-more">
                +${matches.length - 5} more results
              </div>
            `;
          }
          
          html += `
              </div>
            </div>
          `;
        });
      } else {
        html = `
          <div class="suggestion-empty">
            No matching tickers found
          </div>
        `;
        
        // Special handling for potential Indian stocks
        if (!query.includes('.ns') && !query.includes('.bo')) {
          const upperQuery = query.toUpperCase();
          
          // Check if we need to suggest adding .NS or .BO
          if (this.marketData.India && this.marketData.India.exchangeMap && this.marketData.India.exchangeMap[upperQuery]) {
            html += `
              <div class="suggestion-tip">
                Did you mean <b>${this.marketData.India.exchangeMap[upperQuery]}</b>? 
                <a href="#" class="use-suggestion" data-symbol="${this.marketData.India.exchangeMap[upperQuery]}">Use this</a>
              </div>
            `;
          } else {
            html += `
              <div class="suggestion-tip">
                For Indian stocks, try adding .NS for NSE or .BO for BSE (e.g., RELIANCE.NS)
              </div>
            `;
          }
        }
      }
      
      // Add exchange guide at the bottom
      html += `
        <div class="suggestion-guide">
          <div class="guide-title">Exchange Suffix Guide:</div>
          <div class="guide-items">
            <span class="guide-item">India: .NS (NSE), .BO (BSE)</span>
            <span class="guide-item">US: No suffix</span>
            <span class="guide-item">UK: .L</span>
            <span class="guide-item">Germany: .DE</span>
            <span class="guide-item">Hong Kong: .HK</span>
          </div>
        </div>
      `;
      
      // Update suggestions container
      this.suggestionsContainer.innerHTML = html;
      
      // Add click handler for use-suggestion links
      document.querySelectorAll('.use-suggestion').forEach(link => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          this.selectSuggestion(e.target.dataset.symbol);
        });
      });
      
      // Show suggestions
      this.showSuggestionsContainer();
    }
    
    selectSuggestion(symbol) {
      this.searchInput.value = symbol;
      this.hideSuggestions();
    }
    
    showSuggestionsContainer() {
      this.suggestionsContainer.style.display = 'block';
    }
    
    hideSuggestions() {
      this.suggestionsContainer.style.display = 'none';
    }
    
    handleIndianStocks() {
      const value = this.searchInput.value.trim().toUpperCase();
      
      // Skip if already has an exchange suffix
      if (value.includes('.') || value.length === 0) {
        return;
      }
      
      // Check if this is a known Indian stock
      if (this.marketData.India && this.marketData.India.exchangeMap && this.marketData.India.exchangeMap[value]) {
        this.searchInput.value = this.marketData.India.exchangeMap[value];
      }
    }
  }
  
  // Initialize when DOM is loaded
  document.addEventListener('DOMContentLoaded', () => {
    new TickerSearchService();
  });
  