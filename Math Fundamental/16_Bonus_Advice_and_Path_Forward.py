""" Chapter 08: Career Advice and the Path Forward """

# Out main theme is Calculus, Statistics & Probability, Linear Algebra.
# But you should exploring more such as optimization or discrete math if you have some times

""" 1. Redefine Data Science """
# Data science is the analysis of the data to gain actionale insights.
# Combination of statistics, data analysis, data visualization, machine learning, operations research, software engineering.
# Basically, a software engineering with proficientcy in statistics, machine learning, and optimization.

""" 2. A brieft history of data science """

# Around 2000s, the computer generate a lot of informations at hands.
# In 2010, statistician and matimatician will likely to win in the job market.
# In 2012, Harvard declares that "Data scienctist" is the most sexy job in 21st century.
# A lot of company seeking professional in these field. Many job titles larping as "Data Scientist"
# The word "Data-driven" is born on 2013.
# Some science community debunking that data science is not a real science!

# Around 2014, Data was sold as the 'fuel' for crating artificial intelligence.
# Data science professional use Deep learning framework even though they don't know the purposes of it.
# you don't need a science degree to be a Data Sciencetists. BUT.... You must good at math and software engineering!

""" 3. Finding Your Edge """
# Understanding of statistics and machine learning is not enough in the modern world now!
# You will find yourself chasing data sources, engineering scripts and software, scraping documents, scraping excel workbook and even build your own databases.
# It's important to identify and clarify the expectations from your boss that might affect you.
# Some more skills You can learn more to become a better Data Science in modern world:

# 1. SQL Proficientcy
# SQL = Structured Query Language
# A querying language to retrieve, transform and write the data
# A "relational datavbase" is the most common way to organize data, storing data into tables. Mostly found in "MySQL" or "Microsoft SQL Server"
# Businesses use data ware houses and SQL is almosts the means to retrieve the data.

# You can send SQL querry to a databaae from python
# Bring the data back as Pandas DataFrames, Python collections and other structures.

# 16.1: Running an SQL query within Python using SQLLALchemy
from sqlalchemy import inspect, create_engine
from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///thunderbird_manufacturing.db')
inspector = inspect(engine)

# Get all the tables name
print(inspector.get_table_names())

engine = create_engine('sqlite:///thunderbird_manufacturing.db')
conn = engine.connect()

stmt = text("SELECT * FROM CUSTOMER")
results = conn.execute(stmt)

for customer in results:
    print(customer)

# 16.2: Importing an sQL query into a Pandas DataFrame
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('sqlite:///thunderbird_manufacturing.db')
conn = engine.connect()

df = pd.read_sql("SELECT * FROM CUSTOMER", conn)
print(df) # prints SQL results as DataFrame
# Simply put, Pandas and SQL can work together and are not competing technologies.
# NoSQL = "not only SQL" better equipped to store unstructured data.

# 2. Programming Proficiency
# Learn OOP (Objected-Oriented-Programming), functional programming, unit tests, version control (Git and GitHub), Big-O algorithm analysis, cryptography, and other relevant computer science concepts and language that you come arcoss
# There is pressure for data science professionals to now be software engineers.
# Data structures, functional programming, concurency, and other design pattern.

# Learn to solve practical tasks including database, APIs, web services, JSON, parsing, regular expressions, web scraping, securiy and cryptography,
# Cloud computing (AWS, Microsofr Azure), and whatever else helps you become productive in standing up a system.

# ***** Everything code that write here not using Jupyter Notebook but if your employer asks for it use it *****
# ***** We write Python and runs it in the most traditional ways to understand fundamentals of programming *****
# ***** Use Jupyter Notebook BUT don't relie on them that much ******

# 3. Data Visualization
# Be comfortable making charts, graphs, and plots that not only tell stories to management but also help your own data exploration efforts.
# It's okay to use Excel or powerpoint if it's works!
# For Python, mathplotlib, seaborn, sympy, plotly
# The most important things is that depends on the employer preferences.

# 4. Knowing Your Industry
# The risk tolerance gap between both of these industries is wide.
# If you want to do lots of machine learning, you will want to work in low-risk industries where false positves and false nagative do not endanger or upset anyone
# Even with a specialized PhD, false positives and false negatives do not magically disappear

# 5. Productive Learning
# You need to have a project or objective in mind
# Priorities what you learn is an invalueable skill. Only you get to decide what is worth learning.

# 6. Practitioner Versus Advisor
# Manager complainss that their data scientist wants to work on problems they find interesting but that do not add value to the organization
# you have to be knowledgeable and know things other people do not.
# Paying attention to your client's industry as well as other industries, tracking who is succeeding and who is not
# Find the right solutions to the right problem. and be effective communicator and share inforamtion in a way that help client, not just a demonstrate what you know
# The greatest risk for an advisor is providing information that ends up being wrong!

# Profit over People anyday!
# Many founders and investors simply want to rise the growth and cash out before the bubble bursts, of the nwhen the company is sold to the public through an IPO
# recognize what is motivating your client or employer!

# ***** When projects are planned on tools rather than problems, it's gonno not succeeding *****

""" 3. What to looking for in Data Science Jobs """
# No one wants to admit they are out of the loop and ignorant of something important
# "Jabberwocky effect" is an anecdotal theory that an industry or organization can perpetuate a buzzword/project even if no one has satisfactorily defined what it is
# Even tangible, high profile, and specific projects can become mysterious buzzwords understood by a few but talked about by many

# 1. Role Difinition
# 