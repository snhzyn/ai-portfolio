import folium
import hashlib
import json
import psycopg2
from psycopg2.extras import RealDictCursor

"""
Geocoder Module

Provides geospatial utilities for NVISIA, which includs:
- geometry lookup from PostGIS
- spatial join with North Korea organization coordinates
- folium-based article map generation
"""

class Geocoder:
    """
    Geospatial service for article-based map visualization.
    """

    def __init__(self, host, database, user, password, port):
        """        
        Initialize PostgreSQL/PostGIS connection.
        """

        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password, port=port)
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
    
    def get_geometry(self, event_locs):
        """
        Return GeoJSON geometries matched to event locations.
        """

        if not event_locs:
            return {}
        
        query = """
            SELECT event_loc, ST_AsGeoJSON(geometry) AS geojson
            FROM geojson
            WHERE event_loc = ANY(%s)
        """
        self.cur.execute(query, (event_locs,))
        rows = self.cur.fetchall()

        result = {}
        for row in rows:
            result[row["event_loc"]] = json.loads(row["geojson"])
        return result
                         
    def do_spatial_join(self, event_locs, event_orgs):
        """        
        Return organization points located inside selected event location polygons. 
        """
        if not event_locs or not event_orgs:
            return []
        
        query = """
            SELECT
                o.org_name AS org_name,
                g.event_loc,
                ST_Y(ST_Transform(o.geom_5179, 4326)) AS y_4326,
                ST_X(ST_Transform(o.geom_5179, 4326)) AS x_4326
            FROM geojson g
            JOIN nk_org o 
            ON ST_Intersects(ST_Transform(g.geometry, 5179), o.geom_5179)
            WHERE g.event_loc = ANY(%s) AND o.org_name = ANY(%s)
            """
        self.cur.execute(query, (event_locs,event_orgs))
        return self.cur.fetchall()
    
    def _split_event_locs(self, loc_str):
        """
        Split comma-separated event location string into a list.
        """
        return [loc.strip() for loc in loc_str.split(",") if loc.strip()]

    def _split_event_orgs(self, org_str):
        """
        Split comma-separated organization string into a list.
        """
        return [org.strip() for org in org_str.split(",") if org.strip()]

    def extract_event_locs(self, event_loc_series):
        """
        Extract unique event locations from a pandas Series.
        """
        locs = set()
        for loc_str in event_loc_series.fillna(""):
            for loc in self._split_event_locs(loc_str):
                locs.add(loc)
        return sorted(locs)
    
    def extract_event_orgs(self, event_org_series):
        """
        Extract unique organizations from a pandas Series.
        """
        orgs = set()
        for org_str in event_org_series.fillna(""):
            for org in self._split_event_orgs(org_str):
                orgs.add(org)
        return sorted(orgs)   
    
    def _color_from_name(self, name):
        """
        Generate a deterministic color from a name string.
        """
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
        return f"#{h}"

    def build_article_map(self, dataframe, location=(39.0, 127.0), zoom_start=7):
        """
        Build a folium map for the selected article dataframe.
        """
        locs = self.extract_event_locs(dataframe["event_loc"])
        if not locs:
            return None
        
        geo_dict = self.get_geometry(locs)

        m = folium.Map(location=location, zoom_start=zoom_start)

        for loc, geom in geo_dict.items():
            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": {"event_loc": loc},
            }

            folium.GeoJson(
                feature,
                name=loc,
                tooltip=folium.Tooltip(loc),
                style_function=lambda x, loc_name=loc: {
                    "fillColor": self._color_from_name(loc_name),
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.6,
                },
                highlight_function=lambda x: {
                    "weight": 3,
                    "color": "yellow",
                    "fillOpacity": 0.8,
                },
            ).add_to(m)

        event_orgs = self.extract_event_orgs(dataframe["event_org"])
        org_rows = self.do_spatial_join(locs, event_orgs)

        org_fg = folium.FeatureGroup(name="주요 위치", show=True)

        for r in org_rows:
            folium.CircleMarker(
                location=[r["y_4326"], r["x_4326"]],
                radius=3,
                tooltip=r["org_name"],
                fill=True,
                fill_opacity=0.9,
                color="red",
            ).add_to(org_fg)

        org_fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    def close(self):
        self.cur.close()
        self.conn.close()