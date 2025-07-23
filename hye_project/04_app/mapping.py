
borough_map = {
    "Manhattan": [
        # from nbr_grp_04
        "Hell's Kitchen", "Chelsea", "Lower East Side", "East Village",
        "Upper East Side", "Upper West Side", "Chinatown", "Gramercy",
        "Little Italy", "Roosevelt Island",
        # from nbr_grp_03
        "Two Bridges", "East Harlem",
        # from nbr_grp_05
        "Harlem", "Washington Heights", "Maspeth",  # Maspeth 경계상 퀸즈와 접하지만 ManhattanCB5에 일부 포함
        "Morningside Heights",
        # from nbr_grp_01
        "Midtown", "West Village", "Kips Bay", "Nolita",
        "Greenwich Village", "Tribeca", "SoHo", "Murray Hill",
        "Financial District", "Theater District", "Battery Park City",
        "Civic Center", "NoHo", "Flatiron District"
    ],
    "Brooklyn": [
        # nbr_grp_04
        "Prospect Heights", "Williamsburg", "Fort Greene", "Clinton Hill",
        "Gowanus", "Park Slope", "South Slope", "Windsor Terrace",
        "Prospect-Lefferts Gardens", "Downtown Brooklyn",
        "Mill Basin", "Bergen Beach", "Navy Yard", "Gerritsen Beach",
        # nbr_grp_03
        "Bedford-Stuyvesant", "Crown Heights", "Bushwick", "Sheepshead Bay",
        "East New York", "Cypress Hills",
        # nbr_grp_05
        "Bushwick",  # 중복 제거 전후
        # nbr_grp_01
        "Carroll Gardens", "Brooklyn Heights", "Boerum Hill",
        "Red Hook", "DUMBO", "Cobble Hill", "Vinegar Hill", "Columbia St"
    ],
    "Queens": [
        # nbr_grp_04
        "Middle Village", "Long Island City", "Springfield Gardens",
        # nbr_grp_05
        "Astoria", "Ridgewood", "Sunnyside", "Ditmars Steinway",
        "Forest Hills", "Flushing", "Rego Park", "Briarwood",
        "Fresh Meadows", "Holliswood", "Jamaica", "Richmond Hill",
        "Soundview", "Bay Terrace", "College Point", "Little Neck",
        "Ozone Park", "Woodhaven", "St. Albans", "Kew Gardens Hills",
        "Cambria Heights", "Laurelton", "Rosedale", "Arverne",
        "Bayside", "Edgemere", "Far Rockaway", "Neponsit", "Rockaway Park",
        "Bayswater", "Belle Harbor"  # Queens CB14
    ],
    "Bronx": [
        # nbr_grp_04
        "Spuyten Duyvil", "Pelham Bay", "East Morrisania", "University Heights",
        "West Farms",
        # nbr_grp_03
        "Mott Haven", "Eastchester", "Port Morris", "City Island",
        "Bedford-Stuyvesant"  # 경계상 일부 Bronx CB1 포함
        # nbr_grp_05
        "Clason Point", "Kingsbridge", "Allerton", "Norwood",
        "Concourse", "Soundview", "Mount Hope", "Concourse Village",
        "Baychester", "Wakefield", "Mount Eden", "Morrisania",
        "Marble Hill", "Melrose", "Throgs Neck", "Parkchester",
        "Schuylerville", "Belmont", "Morris Heights"
    ],
    "Staten Island": [
        # nbr_grp_04
        "Lighthouse Hill", "New Brighton", "Prince's Bay", "Oakwood",
        "Dongan Hills", "Grymes Hill",
        # nbr_grp_01
        "Willowbrook",
        # other
        "Arrochar", "Annadale", "Arden Heights", "Bay Terrace",
        "Bloomfield", "Bulls Head", "Castleton Corners", "Clifton",
        "Concord", "Eltingville", "Emerson Hill", "Fort Wadsworth",
        "Grant City", "Grasmere", "Great Kills", "Huguenot",
        "Mariners Harbor", "Meiers Corners", "Midland Beach",
        "New Dorp Beach", "New Springville", "Oakwood", "Ocean Breeze",
        "Old Town", "Port Richmond", "Randall Manor", "Rosebank",
        "Seaview", "Shore Acres", "South Beach", "Stapleton",
        "St. George", "Todt Hill", "Tottenville", "West Brighton",
        "Westerleigh", "Woodrow"
    ]
}

amenity_selection_map = [
    # Safety
    "Smoke alarm", "Carbon-monoxide alarm", "Fire extinguisher",
    "First-aid kit", "Exterior cameras",
    # Living
    "Wifi", "Air conditioning", "Heating", "Hot water",
    "Shampoo", "Conditioner","Shower gel", "Body soap", "Bed linens", "towels", "Hair-dryer", "Iron",
    "Washer", "Dryer", "Dedicated workspace", "Pets allowed", "Clothing storage"
    # Kitchen
    "Kitchen", "Cooking basics", "Refrigerator", "Microwave",
    "Oven", "Stove", "Dishwasher", "Coffee maker","Wine glasses", "Toaster", "Dining table",
    # Entertainment
    "TV", "Streaming services", "Sound system / Bluetooth speaker",
    "Board & video games",
    # Outdoor / Facilities
    "Backyard", "Patio / Balcony", "Outdoor furniture", "BBQ grill",
    "Pool", "Bathtub", "Gym", "Free parking", "Paid parking",
    "EV charger", "Elevator", "Lockbox", "Pets allowed", "Self check-in"
]

amenity_group_map={
 'common':['Carbon monoxide alarm','Essentials','Hangers','Smoke alarm','Wifi'],
 'high':['Air conditioning','Building staff','Elevator','Gym','Heating','Paid parking off premises','Shampoo'],
 'low-mid':['Cleaning products','Dining table','Exterior security cameras on property','Free street parking','Freezer','Laundromat nearby','Lock on bedroom door','Microwave'],
 'mid':['Cooking basics','Kitchen','Oven'],
 'upper-mid':['Bathtub','Cleaning products','Cooking basics','Dishes and silverware','Elevator','Freezer']
}

