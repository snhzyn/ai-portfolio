import streamlit as st

from core.config import DB
from app.services.rec import Recommender
from app.services.geocoder import Geocoder

@st.cache_resource
def get_recommender():
    """
    Return a cached Recommender instance.
    """
    return Recommender(**DB)

@st.cache_resource
def get_geocoder():
    """
    Return a cached Geocoder instance.
    """
    return Geocoder(**DB)