import re
import pandas as pd

class LocationNormalizer:
    """
    Normalize extracted location strings using North Korea city/province reference data.
    """

    def __init__(self, nk_cities_path, encoding="euc-kr"):

        self.nk_cities = pd.read_csv(nk_cities_path, encoding=encoding)
        self.provinces_map, self.cities_map = self._build_maps()
        self.BROAD_TERMS_MAP = {
            "평안도": ["평안남도", "평안북도"],
            "함경도": ["함경남도", "함경북도"],
            "황해도": ["황해남도", "황해북도"],
        }

    def _get_search_keys(self, name):
        if pd.isna(name): return [], None
        # "나선시(라선시)" -> ["나선시", "라선시"]
        parts = re.split(r'[()]', name)
        parts = [p.strip() for p in parts if p.strip()]
        
        canonical_name = parts[0]
        
        keys = []
        for p in parts:
            key = p
            if key.endswith('도'): key = key[:-1]
            elif key.endswith('시'): key = key[:-1]
            elif key.endswith('군'): key = key[:-1]
            elif key.endswith('구역'): key = key[:-1]
            keys.append(key)
        return keys, canonical_name

    def _build_maps(self):
        provinces_map = {} 
        cities_map = {}    

        for idx, row in self.nk_cities.iterrows():

            p_keys, p_canon = self._get_search_keys(row['도'])
            for k in p_keys:
                provinces_map[k] = p_canon
                
            c_keys, c_canon = self._get_search_keys(row['시'])
            for k in c_keys:
                cities_map[k] = {
                    'full': c_canon,
                    'province': p_canon 
                }

        abbr_map = {
            '평남': '평안남도',
            '평북': '평안북도',
            '함남': '함경남도',
            '함북': '함경북도',
            '황남': '황해남도',
            '황북': '황해북도',
            '양강': '양강도',
            '자강': '자강도',
            '강원': '강원도',
            '평안도': '평안도',
            '황해도': '황해도', 
            '함경도': '함경도', 
            '평안': '평안도' 
        }

        for abbr, full in abbr_map.items():
            provinces_map[abbr] = full
            
        return provinces_map, cities_map

    def normalize(self, loc_str):
        if pd.isna(loc_str) or not isinstance(loc_str, str):
            return None
        
        found_provinces = set()
        found_cities = [] 
        
        for key, full_name in self.provinces_map.items():
            if key in loc_str:
                found_provinces.add(full_name)
                
        for key, info in self.cities_map.items():
            if key in loc_str:
                match_info = info.copy()
                match_info['key'] = key
                found_cities.append(match_info)
                
        implied_provinces = set()
        for c in found_cities:
            if pd.notna(c['province']):
                implied_provinces.add(c['province'])
                
        temp_provinces = set()
        for p in found_provinces:
            if p not in implied_provinces:
                temp_provinces.add(p)
        
        all_present_specific_provinces = temp_provinces.union(implied_provinces)
        
        final_provinces = set()
        for p in temp_provinces:
            is_redundant_broad = False
            if p in self.BROAD_TERMS_MAP:
                for specific in self.BROAD_TERMS_MAP[p]:
                    if specific in all_present_specific_provinces:
                        is_redundant_broad = True
                        break
            
            if not is_redundant_broad:
                final_provinces.add(p)
                
        final_results = set()
        
        for p in final_provinces:
            final_results.add(p)
            
        for c in found_cities:
            full_city = c['full']
            province = c['province']
            
            if pd.notna(province):
                final_results.add(f"{province} {full_city}")
            else:
                final_results.add(full_city)
                
        if not final_results:
            return None
            
        return ', '.join(sorted(list(final_results)))        