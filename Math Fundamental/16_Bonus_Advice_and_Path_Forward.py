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
# You always want to go into a role that is clearly defined and has tangible objectives.
# Should have leadership with a clear vision that understands what the business needs.
# If the department wants you for being "data-driven" or have a competitive edge in "data science" this is a red flag
# Beware of hiring based on the buzzword

# 2. Organizational Focus and Buy-In
# How aligned the organizations is on specific objectives and whether all parties are bough in
# Management creates a data science team but there's no clear objective

# Leadership must have this:
# 2.1: Have a clearly defined objective and road map
# 2.2: Obtain budget to collect data and support the infrastructure
# 2.3: Attain data access and negotiate data ownership
# 2.4: Include stakeholder buy0in and domain knowledge
# 2.5: Budget time and meeting from stakeholders.

# Data Acess is Political!
# If you are asking for access to their data, you are asking to get into their business

# 3. Adequate Resources
# Another risk to watch out for is not getting adequate resources to do your job.
# It is difficult being thrown into a role and not having what you need.
# Something can cost money and your organization may be unable to budget for them.

# 4. Reasonable Objectives
# The big one to watch out for.
# Your first 18 months is mostly explaining to management why they have not delivered, because they are still trying to gather and clean data, which is 95% of ML efforts.
# Unrealistic expectations.
# They just traded one set of manual processes for another: the procurement of labeled data.
# Find a way to manage expectations with management.

# High-dollar management consultancies create more billable hours around "AI Strategy"
# "LARPing"
# They have no long-term stake in their customer's success.
# Be a guy who selling shovels during a gold rush

# So, Always be aware of this dynamic and always ask "cui bono?" = Who benefits?

# 5. Competing with Existing Systems
# The situation in work environment that lack things to do and need to look busy.
# If the existing system you are competing with is broken and rudimentary or done completely manually without automation?
# Run fast!

# A bullshit job is paid employment that is so completely pointless, unnecessary or pernicious that even the employee cannot justify its exitence but has to pretend otherwise.
# If you find yourself in a job not designed to create value, think strategically how you can influence postive change, or find a new opportunities.

# 6. A Role Is NOT what you think
# You can certainly accept your data scientist role turning into an IT role, and maybe get some database and programming skills in the process!
# Become an advocate for change. Push to modernize the tooling, advocating using Python and a modern database platform like MySQL or SQLite.
# Nasty politics can ensure when IT departments and non-IT departments clash, accusing each other of not staying in their lane or simply co-opting roles for job security.
# Overall, If you find yourself in a shadow IT role, make sure you understand the risks and play nice with the IT department.

# 7. Does Your Dream Job Not Exist?
# Always check up your expectations too.
# Keep your expectation realistic as you navigate the job market!
# A data sciencetist is evolving to be a software engineer with proficiency in statistics, optimization, and ML
# Consider others Role like: Computer Vision Engineer, Data Engineer, Data Analyst, Researcher, Operations Research Analyst, and Advisor/Consultant, or even a quants!

# Summary:
# 1. Pick a small piece of something and start there.
# 2. Do something small. Learn something small. Build something small. Every single day!
# 3. Understand why things work, not just how they work.
# 4. Learn so you can match the right tools to the right problems and find the right solutions.
# 5. Let it compounds. Never Stop learning and improving yourself.

# Bonus - 16.3: Using SymPy to convert an expression into LaTex
# you gonna use math a lot. I recommend using LaTex to write a high quality papers and equations on it.
from sympy import *

x, y = symbols('x y')
z = x ** 2 / sqrt(2 * y ** 3 - 1)
print(latex(z)) # \frac{x^{2}}{\sqrt{2 y^{3} - 1}}
# You gonna write some research papers some day. So using LaTex is a great choice for producing a clear formulas and equations!

# The End.