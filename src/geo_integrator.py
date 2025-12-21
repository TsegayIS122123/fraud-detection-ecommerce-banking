"""
Geolocation Integration Module - OPTIMIZED VERSION with pandas merge_asof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GeoIntegrator:
    """Integrates geolocation data by mapping IP addresses to countries - OPTIMIZED."""
    
    def __init__(self):
        self.geo_report = {}
    
    def ip_to_int_fast(self, ip_series):
        """Convert IP address series to integer array - VECTORIZED for speed."""
        def ip_to_int_single(ip):
            if pd.isna(ip):
                return 0
            try:
                octets = list(map(int, str(ip).split('.')))
                return (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
            except:
                return 0
        
        # Use vectorized operation if possible, otherwise apply
        try:
            # Vectorized version for speed
            split_ips = ip_series.str.split('.', expand=True).astype(float)
            return (split_ips[0] * 16777216 + 
                   split_ips[1] * 65536 + 
                   split_ips[2] * 256 + 
                   split_ips[3]).fillna(0).astype(np.int64)
        except:
            # Fallback to apply
            return ip_series.apply(ip_to_int_single)
    
    def merge_with_country_data(self, fraud_df, ip_country_df, ip_col='ip_address', method='fast'):
        """
        Merge fraud data with country data using specified method.
        
        Parameters:
        -----------
        fraud_df : pd.DataFrame
            Fraud transaction data
        ip_country_df : pd.DataFrame
            IP to country mapping data
        ip_col : str
            Column name containing IP addresses
        method : str
            'fast' for merge_asof, 'simple' for demo version
        
        Returns:
        --------
        pd.DataFrame
            Fraud data with added 'country' column
        """
        if method == 'fast':
            return self._merge_fast(fraud_df, ip_country_df, ip_col)
        elif method == 'simple':
            return self._merge_simple(fraud_df, ip_country_df, ip_col)
        else:
            print(f"Unknown method: {method}. Using fast method.")
            return self._merge_fast(fraud_df, ip_country_df, ip_col)
    
    def _merge_fast(self, fraud_df, ip_country_df, ip_col='ip_address'):
        """FAST VERSION: Merge using pandas merge_asfor O(n log n) complexity."""
        print(f"\n{'='*60}")
        print("GEOLOCATION INTEGRATION: FAST VERSION with pandas merge_asof")
        print(f"{'='*60}")
        
        import time
        start_time = time.time()
        
        # Step 1: Convert IPs to integers - VECTORIZED
        print("\n1. Converting IP addresses to integers (vectorized)...")
        fraud_df = fraud_df.copy()
        fraud_df['ip_int'] = self.ip_to_int_fast(fraud_df[ip_col])
        
        # Step 2: Prepare IP country data
        print("2. Preparing IP country mapping data...")
        ip_country_df = ip_country_df.copy()
        ip_country_df['lower_bound_ip_address'] = pd.to_numeric(ip_country_df['lower_bound_ip_address'])
        ip_country_df['upper_bound_ip_address'] = pd.to_numeric(ip_country_df['upper_bound_ip_address'])
        
        # Create a mapping table with just lower bounds and countries
        mapping_df = ip_country_df[['lower_bound_ip_address', 'country']].copy()
        mapping_df = mapping_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
        
        # Step 3: Sort fraud data by IP integer
        print("3. Sorting data for fast merge...")
        fraud_sorted = fraud_df.sort_values('ip_int').reset_index(drop=True)
        
        # Step 4: Use pandas merge_asof for O(n log n) merge
        print("4. Performing fast merge using merge_asof...")
        merged_df = pd.merge_asof(
            fraud_sorted,
            mapping_df,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        
        # Step 5: Handle unmatched IPs
        print("5. Handling unmatched IP addresses...")
        merged_df['country'] = merged_df['country'].fillna('Unknown')
        
        # Step 6: Restore original order
        merged_df = merged_df.sort_index()
        
        # Step 7: Analyze results
        unique_countries = merged_df['country'].nunique()
        unknown_ips = (merged_df['country'] == 'Unknown').sum()
        elapsed_time = time.time() - start_time
        
        print("\n6. INTEGRATION RESULTS:")
        print("-" * 40)
        print(f"Total transactions: {len(merged_df):,}")
        print(f"Unique countries mapped: {unique_countries}")
        print(f"IPs with unknown country: {unknown_ips} ({unknown_ips/len(merged_df)*100:.2f}%)")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Speed: {len(merged_df)/elapsed_time:,.0f} transactions/second")
        
        # Store in report
        self.geo_report = {
            'total_transactions': len(merged_df),
            'unique_countries': unique_countries,
            'unknown_ip_count': unknown_ips,
            'unknown_ip_percent': unknown_ips/len(merged_df)*100,
            'processing_time_seconds': elapsed_time,
            'speed_txn_per_second': len(merged_df)/elapsed_time
        }
        
        # Drop temporary columns
        merged_df = merged_df.drop(columns=['ip_int', 'lower_bound_ip_address'], errors='ignore')
        
        return merged_df
    
    def _merge_simple(self, fraud_df, ip_country_df, ip_col='ip_address', sample_size=None):
        """SIMPLE VERSION: For testing or when full mapping isn't needed."""
        print(f"\n{'='*60}")
        print("GEOLOCATION INTEGRATION: SIMPLE VERSION")
        print(f"{'='*60}")
        
        fraud_df = fraud_df.copy()
        
        if sample_size and sample_size < len(fraud_df):
            print(f"Using sample of {sample_size:,} transactions...")
            fraud_df = fraud_df.sample(n=sample_size, random_state=42)
        
        # Create simplified country mapping based on IP first octet
        print("Creating simplified country mapping based on IP patterns...")
        
        # Extract first octet of IP
        def get_ip_prefix(ip):
            if pd.isna(ip):
                return '0'
            try:
                return ip.split('.')[0]
            except:
                return '0'
        
        fraud_df['ip_prefix'] = fraud_df[ip_col].apply(get_ip_prefix)
        
        # Create a mock country mapping based on IP prefix
        country_mapping = {
            '1': 'Country_A', '2': 'Country_B', '3': 'Country_C',
            '4': 'Country_D', '5': 'Country_E', '6': 'Country_F',
            '7': 'Country_G', '8': 'Country_H', '9': 'Country_I',
            '10': 'Country_J', '11': 'Country_K', '12': 'Country_L'
        }
        
        fraud_df['country'] = fraud_df['ip_prefix'].map(country_mapping).fillna('Other')
        
        print(f"\n✅ Simplified geolocation complete!")
        print(f"Added 'country' column with {fraud_df['country'].nunique()} unique countries")
        print(f"Using IP prefix mapping for demonstration")
        
        return fraud_df.drop(columns=['ip_prefix'], errors='ignore')
    
    def analyze_fraud_by_country(self, fraud_df_with_country, target_col='class', top_n=15):
        """Analyze fraud patterns by country."""
        print(f"\n{'='*60}")
        print("FRAUD PATTERN ANALYSIS BY COUNTRY")
        print(f"{'='*60}")
        
        # Calculate fraud statistics by country
        fraud_by_country = fraud_df_with_country.groupby('country').agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        
        fraud_by_country.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        fraud_by_country['fraud_rate_percent'] = fraud_by_country['fraud_rate'] * 100
        
        # Sort by fraud rate (descending)
        fraud_by_country = fraud_by_country.sort_values('fraud_rate', ascending=False)
        
        print(f"\nTop {top_n} Countries by Fraud Rate:")
        print("-" * 50)
        print(fraud_by_country.head(top_n).to_string())
        
        print(f"\nBottom {top_n} Countries by Fraud Rate:")
        print("-" * 50)
        print(fraud_by_country.tail(top_n).to_string())
        
        # Summary statistics
        print(f"\nCOUNTRY FRAUD SUMMARY:")
        print("-" * 40)
        print(f"Total countries analyzed: {len(fraud_by_country)}")
        print(f"Average fraud rate: {fraud_by_country['fraud_rate'].mean():.4%}")
        print(f"Median fraud rate: {fraud_by_country['fraud_rate'].median():.4%}")
        print(f"Std deviation of fraud rates: {fraud_by_country['fraud_rate'].std():.4%}")
        
        if len(fraud_by_country) > 0:
            print(f"Country with highest fraud rate: {fraud_by_country.index[0]} ({fraud_by_country.iloc[0]['fraud_rate']:.4%})")
            print(f"Country with lowest fraud rate: {fraud_by_country.index[-1]} ({fraud_by_country.iloc[-1]['fraud_rate']:.4%})")
        
        # Store in report
        self.geo_report['fraud_by_country'] = fraud_by_country.head(20).to_dict()
        if len(fraud_by_country) > 0:
            self.geo_report['top_country'] = fraud_by_country.index[0]
            self.geo_report['top_fraud_rate'] = fraud_by_country.iloc[0]['fraud_rate']
        
        return fraud_by_country
    
    def get_geo_report(self):
        """Return geolocation integration report."""
        return self.geo_report