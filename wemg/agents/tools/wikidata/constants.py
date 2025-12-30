"""Constants and configuration for Wikidata tools."""

# Query limits
WIKIDATA_MAX_QUERY_LENGTH = 500

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = 4  # Maximum concurrent requests to Wikidata
REQUEST_DELAY = 0.1  # Base delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_BASE_DELAY = 10  # Base delay for exponential backoff
USER_AGENT = "WEMG-Bot/1.0 (https://github.com/uonlp/wemg; contact@example.com) Python/3.x"

# Batch processing sizes
BATCH_SIZE_ENTITY_SEARCH = 25
BATCH_SIZE_ENTITY_RETRIEVAL = 25
BATCH_SIZE_PROPERTY_RETRIEVAL = 50
BATCH_SIZE_TRIPLE_QUERY = 10
LIMIT_PER_QUERY = 100
MAX_ENTITIES_PER_HOP = 1000

# Default properties to retrieve
DEFAULT_PROPERTIES = [
    "P27",
    "P361",
    "P527",
    "P495",
    "P17",
    "P585",
    "P106",
    "P569",
    "P570",
    "P577",
    "P50",
    "P571",
    "P641",
    "P625",
    "P19",
    "P69",
    "P108",
    "P136",
    "P39",
    "P161",
    "P20",
    "P101",
    "P179",
    "P175",
    "P7937",
    "P57",
    "P607",
    "P509",
    "P800",
    "P449",
    "P580",
    "P582",
    "P276",
    "P112",
    "P740",
    "P159",
    "P452",
    "P102",
    "P1142",
    "P1387",
    "P1576",
    "P140",
    "P178",
    "P287",
    "P25",
    "P22",
    "P40",
    "P185",
    "P802",
    "P1416",
    "P26",
    "P3373",
    # People - Relationships
    "P451",
    "P1038",
    # People - Education/Career
    "P184",
    "P166",
    "P512",
    # Organizations
    "P463",
    "P127",
    "P749",
    "P355",
    "P488",
    "P169",
    # Geography/Location
    "P131",
    "P706",
    "P150",
    "P36",
    "P30",
    # Government/Politics
    "P6",
    "P35",
    "P1313",
    # Sports
    "P54",
    "P1344",
    "P1532",
    # Creative Works
    "P170",
    "P86",
    "P162",
    "P58",
    "P144",
    "P921",
    "P407",
]

# Property labels and descriptions
PROPERTY_LABELS = {
    'P1142': {
        'label': 'political ideology',
        'description': 'political ideology of an organization or person or of a work (such as a newspaper)'
    },
    'P69': {
        'label': 'educated at',
        'description': 'educational institution attended by subject'
    },
    'P108': {
        'label': 'employer',
        'description': 'person or organization for which the subject works or worked'
    },
    'P136': {
        'label': 'genre',
        'description': "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic"
    },
    'P50': {
        'label': 'author',
        'description': 'main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist'
    },
    'P22': {
        'label': 'father',
        'description': 'male parent of the subject. For stepfather, use "stepparent" (P3448)'
    },
    'P19': {
        'label': 'place of birth',
        'description': 'most specific known birth location of a person, animal or fictional character'
    },
    'P112': {
        'label': 'founded by',
        'description': 'founder or co-founder of this organization, religion, place or entity'
    },
    'P495': {
        'label': 'country of origin',
        'description': 'country of origin of this item (creative work, food, phrase, product, etc.)'
    },
    'P27': {
        'label': 'country of citizenship',
        'description': 'the object is a country that recognizes the subject as its citizen'
    },
    'P509': {
        'label': 'cause of death',
        'description': "underlying or immediate cause of death. Underlying cause (e.g. car accident, stomach cancer) preferred. Use 'manner of death' (P1196) for broadest category, e.g. natural causes, accident, homicide, suicide"
    },
    'P287': {
        'label': 'designed by',
        'description': 'person or organization which designed the object. For buildings use "architect" (Property:P84)'
    },
    'P159': {
        'label': 'headquarters location',
        'description': "city or town where an organization's headquarters is or has been situated. Use P276 qualifier for specific building"
    },
    'P571': {
        'label': 'inception',
        'description': 'time when an entity begins to exist; for date of official opening use P1619'
    },
    'P25': {
        'label': 'mother',
        'description': 'female parent of the subject. For stepmother, use "stepparent" (P3448)'
    },
    'P106': {
        'label': 'occupation',
        'description': 'occupation of a person. See also "field of work" (Property:P101), "position held" (Property:P39). Not for groups of people. There, use "field of work" (Property:P101), "industry" (Property:P452), "members have occupation" (Property:P3989).'
    },
    'P57': {
        'label': 'director',
        'description': 'director(s) of film, TV-series, stageplay, video game or similar'
    },
    'P102': {
        'label': 'member of political party',
        'description': 'the political party of which a person is or has been a member or otherwise affiliated'
    },
    'P607': {
        'label': 'participated in conflict',
        'description': 'battles, wars or other military engagements in which the person or item participated'
    },
    'P179': {
        'label': 'part of the series',
        'description': 'series which contains the subject'
    },
    'P40': {
        'label': 'child',
        'description': 'subject has object as child. Do not use for stepchildren—use "relative" (P1038), qualified with "type of kinship" (P1039)'
    },
    'P1416': {
        'label': 'affiliation',
        'description': 'organization that a person or organization is affiliated with (not necessarily member of or employed by)'
    },
    'P361': {
        'label': 'part of',
        'description': 'object of which the subject is a part (if this subject is already part of object A which is a part of object B, then please only make the subject part of object A), inverse property of "has part" (P527, see also "has parts of the class" (P2670))'
    },
    'P527': {
        'label': 'has part(s)',
        'description': 'part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).'
    },
    'P449': {
        'label': 'original broadcaster',
        'description': 'network(s) or service(s) that originally broadcast a radio or television program'
    },
    'P101': {
        'label': 'field of work',
        'description': 'specialization of a person or organization; see P106 for the occupation'
    },
    'P7937': {
        'label': 'form of creative work',
        'description': 'structure of a creative work'
    },
    'P178': {
        'label': 'developer',
        'description': 'organization or person that developed the item'
    },
    'P17': {
        'label': 'country',
        'description': 'sovereign state that this item is in (not to be used for human beings)'
    },
    'P39': {
        'label': 'position held',
        'description': 'subject currently or formerly holds the object position or public office'
    },
    'P175': {
        'label': 'performer',
        'description': 'actor, musician, band or other performer associated with this role or musical work'
    },
    'P585': {
        'label': 'point in time',
        'description': 'date something took place, existed or a statement was true; for providing time use the "refine date" property (P4241)'
    },
    'P577': {
        'label': 'publication date',
        'description': 'date or point in time when a work or product was first published or released'
    },
    'P802': {
        'label': 'student',
        'description': 'notable student(s) of the subject individual'
    },
    'P140': {
        'label': 'religion or worldview',
        'description': 'religion of a person, organization or religious building, or associated with this subject'
    },
    'P740': {
        'label': 'location of formation',
        'description': 'location where a group or organization was formed'
    },
    'P20': {
        'label': 'place of death',
        'description': 'most specific known (e.g. city instead of country, or hospital instead of city) death location of a person, animal or fictional character'
    },
    'P800': {
        'label': 'notable work',
        'description': "notable scientific, artistic or literary work, or other work of significance among subject's works"
    },
    'P570': {
        'label': 'date of death',
        'description': 'date on which the subject died'
    },
    'P580': {
        'label': 'start time',
        'description': 'time an entity begins to exist or a statement starts being valid'
    },
    'P582': {
        'label': 'end time',
        'description': 'moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true'
    },
    'P569': {
        'label': 'date of birth',
        'description': 'date on which the subject was born'
    },
    'P276': {
        'label': 'location',
        'description': 'location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object'
    },
    'P1387': {
        'label': 'political alignment',
        'description': 'political position within the left–right political spectrum'
    },
    'P185': {
        'label': 'doctoral student',
        'description': 'doctoral student(s) of a professor'
    },
    'P641': {
        'label': 'sport',
        'description': 'sport that the subject participates or participated in or is associated with'
    },
    'P452': {
        'label': 'industry',
        'description': 'specific industry of company or organization'
    },
    'P625': {
        'label': 'coordinate location',
        'description': 'geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported'
    },
    'P1576': {
        'label': 'lifestyle',
        'description': 'typical way of life of an individual, group, or culture'
    },
    'P161': {
        'label': 'cast member',
        'description': 'actor in the subject production [use "character role" (P453) and/or "name of the character role" (P4633) as qualifiers] [use "voice actor" (P725) for voice-only role] - [use "recorded participant" (P11108) for non-fiction productions]'
    },
    'P26': {
        'label': 'spouse',
        'description': 'the subject has the object as their spouse (husband, wife, partner, etc.). Use "unmarried partner" (P451) for non-married companions'
    },
    'P3373': {
        'label': 'sibling',
        'description': 'the subject and the object have at least one common parent (brother, sister, etc. including half-siblings); use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)'
    },
    # People - Relationships
    'P451': {
        'label': 'unmarried partner',
        'description': 'someone with whom the person is in a relationship without being married. Use "spouse" (P26) for married couples'
    },
    'P1038': {
        'label': 'relative',
        'description': 'family member (qualify with "kinship to subject", P1039; for direct family member please use specific property)'
    },
    # People - Education/Career
    'P184': {
        'label': 'doctoral advisor',
        'description': 'person who supervised the doctorate or PhD thesis of the subject'
    },
    'P166': {
        'label': 'award received',
        'description': 'award or recognition received by a person, organization or creative work'
    },
    'P512': {
        'label': 'academic degree',
        'description': 'academic degree that the person holds'
    },
    # Organizations
    'P463': {
        'label': 'member of',
        'description': 'organization, club or musical group to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a political position, such as a member of parliament (use P39 for that)'
    },
    'P127': {
        'label': 'owned by',
        'description': 'owner of the subject'
    },
    'P749': {
        'label': 'parent organization',
        'description': 'parent organization or unit of an organization or unit, opposite of child organization or unit (P355)'
    },
    'P355': {
        'label': 'has subsidiary',
        'description': 'child organization/unit of an organization/unit; for companies, generally a fully owned separate corp., opposite of parent org./unit (P749)'
    },
    'P488': {
        'label': 'chairperson',
        'description': 'presiding member of an organization, group or body'
    },
    'P169': {
        'label': 'chief executive officer',
        'description': 'highest-ranking corporate officer appointed as the CEO within an organization'
    },
    # Geography/Location
    'P131': {
        'label': 'located in the administrative territorial entity',
        'description': 'the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events'
    },
    'P706': {
        'label': 'located in/on physical feature',
        'description': 'located on the specified (geo)physical feature. Should not be used when the value is only political/administrative (P131) or a mountain range (P4552)'
    },
    'P150': {
        'label': 'contains the administrative territorial entity',
        'description': '(list of) direct subdivisions of an administrative territorial entity'
    },
    'P36': {
        'label': 'capital',
        'description': 'seat of government of a country, province, state or other type of administrative territorial entity'
    },
    'P30': {
        'label': 'continent',
        'description': 'continent of which the subject is a part'
    },
    # Government/Politics
    'P6': {
        'label': 'head of government',
        'description': 'head of the executive power of this town, city, municipality, state, country, or other governmental body'
    },
    'P35': {
        'label': 'head of state',
        'description': 'official with the highest formal authority in a country/state'
    },
    'P1313': {
        'label': 'office held by head of government',
        'description': 'political office that is fulfilled by the head of the government of this item'
    },
    # Sports
    'P54': {
        'label': 'member of sports team',
        'description': 'sports teams or clubs that the subject represents or represented'
    },
    'P1344': {
        'label': 'participant in',
        'description': 'event in which a person, organization or creative work was/is a participant; inverse of P710 or P1923'
    },
    'P1532': {
        'label': 'country for sport',
        'description': 'country a person or a team represents when playing a sport'
    },
    # Creative Works
    'P170': {
        'label': 'creator',
        'description': 'maker of this creative work or other object (where no more specific property exists)'
    },
    'P86': {
        'label': 'composer',
        'description': 'person(s) who wrote the music [for lyricist, use "lyrics by" (P676)]'
    },
    'P162': {
        'label': 'producer',
        'description': 'person(s) who produced the film, musical work, theatrical production, etc. (for film, this does not include executive producers, associate producers, etc.)'
    },
    'P58': {
        'label': 'screenwriter',
        'description': 'person(s) who wrote the script for subject item'
    },
    'P144': {
        'label': 'based on',
        'description': 'the work(s) or inputs used as the basis for subject item; for fictional analog use P1074'
    },
    'P921': {
        'label': 'main subject',
        'description': 'primary topic of a work or act of communication'
    },
    'P407': {
        'label': 'language of work or name',
        'description': 'language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name'
    },
}

