"""SmartQuiz AI — single-file FastAPI final project.

Run locally:
    pip install -r requirements.txt
    export JWT_SECRET="replace-with-a-long-random-secret"
    uvicorn main:app --reload

Or run this file directly without a reloader:
    python main.py

The application uses SQLite by default and creates ``smartquiz_single_file.db``
automatically. Set SMARTQUIZ_DATABASE_URL to use another SQLAlchemy database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("SMARTQUIZ_DATABASE_URL", "sqlite:///./smartquiz_single_file.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-demo-secret-before-submission-please")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 60 * 24

engine_args = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
bearer = HTTPBearer(auto_error=False)


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    grade: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Quiz(Base):
    __tablename__ = "quizzes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    grade: Mapped[int] = mapped_column(Integer)
    subject: Mapped[str] = mapped_column(String(120))
    topics: Mapped[list[str]] = mapped_column(JSON)
    mode: Mapped[str] = mapped_column(String(20), default="personalized")
    timer_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    adaptive_difficulty: Mapped[str] = mapped_column(String(10), default="medium")
    status: Mapped[str] = mapped_column(String(20), default="active")
    score_percent: Mapped[float | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"), index=True)
    source_question_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    topic: Mapped[str] = mapped_column(String(180))
    prompt: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(JSON)
    correct_index: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[str] = mapped_column(Text)
    difficulty: Mapped[str] = mapped_column(String(10))
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    selected_index: Mapped[int] = mapped_column(Integer)
    is_correct: Mapped[bool] = mapped_column(Boolean)
    answered_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    reminder_time: Mapped[str] = mapped_column(String(5), default="18:00")
    timezone: Mapped[str] = mapped_column(String(80), default="UTC")
    last_dismissed_on: Mapped[str | None] = mapped_column(String(10), nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Validation schemas
# ---------------------------------------------------------------------------

Difficulty = Literal["easy", "medium", "hard"]


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    grade: int = Field(ge=9, le=12)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    grade: int


class QuizCreateIn(BaseModel):
    grade: int = Field(ge=9, le=12)
    subject: str = Field(min_length=2, max_length=120)
    topics: list[str] = Field(min_length=1, max_length=5)
    question_count: int = Field(default=5, ge=3, le=30)
    starting_difficulty: Difficulty = "medium"
    timer_minutes: int | None = Field(default=None, ge=1, le=180)


class QuestionOut(BaseModel):
    id: int
    topic: str
    prompt: str
    options: list[str]
    difficulty: Difficulty


class QuizOut(BaseModel):
    id: int
    grade: int
    subject: str
    topics: list[str]
    mode: str
    timer_minutes: int | None
    expires_at: datetime | None
    adaptive_difficulty: Difficulty
    status: str
    score_percent: float | None
    current_question: QuestionOut | None


class AnswerIn(BaseModel):
    question_id: int
    selected_index: int = Field(ge=0, le=3)


class AnswerOut(BaseModel):
    is_correct: bool
    explanation: str
    adaptive_difficulty: Difficulty
    next_question: QuestionOut | None
    completed: bool


class ReminderIn(BaseModel):
    enabled: bool
    reminder_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(default="UTC", min_length=1, max_length=80)


class ReminderOut(ReminderIn):
    show_today: bool
    message: str | None


# ---------------------------------------------------------------------------
# Subjects and topics for grades 9–12 (theory and practical)
# ---------------------------------------------------------------------------

CATALOG: dict[int, dict[str, list[str]]] = {
    9: {
        "Mathematics": ["Number Systems", "Algebraic Expressions", "Linear Equations", "Geometry"],
        "Physics": ["Motion", "Force", "Work and Energy", "Matter"],
        "Chemistry": ["Atomic Structure", "Elements", "Compounds", "States of Matter"],
        "Biology": ["Cell Biology", "Tissues", "Nutrition", "Ecosystems"],
        "English": ["Grammar", "Comprehension", "Writing", "Vocabulary"],
        "Computer Science": ["Algorithms", "Data Representation", "Programming Basics", "Networks"],
        "Computer Practical": ["Python Variables", "Input and Output", "Conditions", "Loops"],
        "Science Practical": ["Lab Safety", "Measurement", "Data Tables", "Graphs"],
    },
    10: {
        "Mathematics": ["Quadratic Equations", "Trigonometry", "Coordinate Geometry", "Statistics"],
        "Physics": ["Electricity", "Light", "Sound", "Magnetism"],
        "Chemistry": ["Chemical Reactions", "Acids and Bases", "Metals", "Carbon Chemistry"],
        "Biology": ["Life Processes", "Heredity", "Evolution", "Environment"],
        "English": ["Literature", "Essay Writing", "Grammar", "Speaking Skills"],
        "Computer Science": ["Problem Solving", "Databases", "Cyber Safety", "Programming"],
        "Computer Practical": ["Functions", "Lists", "File Handling", "Debugging"],
        "Science Practical": ["Experiment Design", "Variables", "Observation", "Lab Reports"],
    },
    11: {
        "Mathematics": ["Sets", "Functions", "Limits", "Permutations and Combinations"],
        "Physics": ["Kinematics", "Dynamics", "Waves", "Thermodynamics"],
        "Chemistry": ["Mole Concept", "Chemical Bonding", "Equilibrium", "Organic Chemistry"],
        "Biology": ["Cell Division", "Human Physiology", "Genetics", "Plant Biology"],
        "Computer Science": ["Object-Oriented Programming", "Data Structures", "Databases", "Boolean Logic"],
        "Physics Practical": ["Error Analysis", "Mechanics Lab", "Electricity Lab", "Graphing"],
        "Chemistry Practical": ["Titration", "Salt Analysis", "Molarity", "Lab Calculations"],
        "Biology Practical": ["Microscopy", "Specimen Study", "Enzymes", "Experimental Design"],
        "Computer Practical": ["Python Classes", "SQL Queries", "Stacks", "Testing"],
    },
    12: {
        "Mathematics": ["Integration", "Differentiation", "Probability", "Vectors"],
        "Physics": ["Electromagnetism", "Modern Physics", "Electronics", "Optics"],
        "Chemistry": ["Electrochemistry", "Kinetics", "Polymers", "Analytical Chemistry"],
        "Biology": ["Molecular Biology", "Biotechnology", "Evolution", "Ecology"],
        "Computer Science": ["Algorithms", "Operating Systems", "Web Development", "Data Security"],
        "Physics Practical": ["Semiconductor Lab", "Optics Lab", "Oscilloscope", "Data Analysis"],
        "Chemistry Practical": ["Organic Analysis", "Electrochemical Cells", "Kinetics Lab", "Titration"],
        "Biology Practical": ["DNA Extraction", "Ecology Sampling", "Slide Preparation", "Biotechnology"],
        "Computer Practical": ["Python Projects", "SQL Database", "Algorithms", "Documentation"],
    },
}


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, expected = stored.split("$", 1)
    except ValueError:
        return False
    calculated = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000).hex()
    return hmac.compare_digest(calculated, expected)


def create_token(user_id: int) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": expires}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication is required.")
    try:
        user_id = int(jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account was not found.")
    return user


# ---------------------------------------------------------------------------
# Quiz generation and scoring helpers
# ---------------------------------------------------------------------------

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

# The local bank keeps the app useful without an AI key. These are not copies of
# the same generic question: each topic receives a related idea and rotating
# question style. The optional AI route below can still replace these with richer
# generated questions when provider credentials are supplied.
TOPIC_IDEAS = {
    "Number Systems": "real numbers include rational and irrational numbers",
    "Algebraic Expressions": "like terms can be combined only when their variable parts match",
    "Linear Equations": "the same operation must be applied to both sides to preserve equality",
    "Geometry": "properties of shapes can be justified using definitions and theorems",
    "Quadratic Equations": "a quadratic equation can be solved by factorising, completing the square, or using a formula",
    "Trigonometry": "sine, cosine, and tangent connect angles with side ratios in right-angled triangles",
    "Coordinate Geometry": "coordinates locate points and support calculations of gradient, distance, and equations",
    "Statistics": "data should be represented and interpreted using suitable measures and graphs",
    "Sets": "intersection contains elements shared by both sets",
    "Functions": "each valid input is assigned exactly one output",
    "Limits": "a value a function approaches near a chosen input",
    "Permutations and Combinations": "permutations use order while combinations do not",
    "Integration": "accumulation can be represented by the area under a curve",
    "Differentiation": "a derivative measures an instantaneous rate of change",
    "Probability": "probability ranges from 0 for impossible to 1 for certain",
    "Vectors": "a vector has both magnitude and direction",
    "Kinematics": "motion can be described with displacement, velocity, and acceleration",
    "Dynamics": "net force produces acceleration according to Newton's second law",
    "Waves": "wave speed equals frequency multiplied by wavelength",
    "Thermodynamics": "energy is conserved while heat flows from hotter to cooler objects",
    "Electromagnetism": "a changing magnetic field can induce an electric current",
    "Modern Physics": "quantum-scale phenomena do not always follow classical models",
    "Electronics": "components such as resistors and diodes control current in circuits",
    "Optics": "light changes direction when it moves between materials with different refractive indices",
    "Mole Concept": "one mole represents Avogadro's number of particles",
    "Chemical Bonding": "bonding involves electrostatic attraction that stabilizes atoms",
    "Equilibrium": "forward and reverse reaction rates are equal at dynamic equilibrium",
    "Organic Chemistry": "carbon compounds are organised by functional groups and structure",
    "Electrochemistry": "redox reactions transfer electrons and can produce electrical energy",
    "Kinetics": "reaction rate depends on factors such as concentration and temperature",
    "Polymers": "polymers are large molecules built from repeating smaller units",
    "Analytical Chemistry": "measurements identify or quantify chemical substances",
    "Cell Division": "mitosis produces genetically similar cells for growth and repair",
    "Human Physiology": "body systems coordinate to maintain stable internal conditions",
    "Genetics": "genes carry inherited information encoded in DNA",
    "Plant Biology": "photosynthesis converts light energy into stored chemical energy",
    "Molecular Biology": "DNA information is expressed through RNA and proteins",
    "Biotechnology": "living systems can be used to develop useful products or processes",
    "Ecology": "organisms interact with each other and with their environment",
    "Object-Oriented Programming": "objects combine data with methods that operate on that data",
    "Data Structures": "a structure organises data for efficient storage and retrieval",
    "Databases": "tables store related records that can be queried using structured commands",
    "Boolean Logic": "logical operators combine true and false conditions",
    "Algorithms": "an algorithm is a precise sequence of steps to solve a problem",
    "Operating Systems": "an operating system manages hardware resources and running programs",
    "Web Development": "web applications combine a client interface with server-side logic",
    "Data Security": "confidential data should be protected through access control and safe handling",
    "Python Classes": "a class is a blueprint used to create related objects",
    "SQL Queries": "a SELECT query retrieves chosen data from a database table",
    "Stacks": "a stack follows last-in, first-out order",
    "Testing": "tests compare actual program behavior with an expected result",
    "Lab Safety": "safe practical work uses protective equipment and follows written procedures",
    "Titration": "a known concentration is used to determine an unknown concentration",
    "Microscopy": "magnification helps make small structures visible for observation",
    "DNA Extraction": "cells are opened so DNA can be separated and observed",
    "Motion": "speed describes distance travelled per unit time while velocity includes direction",
    "Force": "a force can change an object's motion or shape",
    "Work and Energy": "work is done when a force causes displacement and energy is conserved",
    "Matter": "particles in solids, liquids, and gases differ in arrangement and movement",
    "Electricity": "current is the rate of flow of electric charge",
    "Light": "light can reflect, refract, and form images",
    "Sound": "sound is produced by vibrations and travels as a wave through a medium",
    "Magnetism": "magnetic fields exert forces on suitable materials and moving charges",
    "Atomic Structure": "atoms contain protons and neutrons in a nucleus with electrons outside it",
    "Elements": "an element contains only one type of atom",
    "Compounds": "a compound contains elements chemically combined in fixed proportions",
    "States of Matter": "particle energy and arrangement explain changes of state",
    "Chemical Reactions": "chemical reactions rearrange atoms while conserving total mass",
    "Acids and Bases": "acids and bases can be identified by their chemical behaviour and pH",
    "Metals": "metal properties relate to their structure and bonding",
    "Carbon Chemistry": "carbon forms a wide range of compounds through covalent bonding",
    "Cell Biology": "cells are the basic units of structure and function in living things",
    "Tissues": "similar specialised cells work together as tissues",
    "Nutrition": "organisms require nutrients for energy, growth, and repair",
    "Ecosystems": "energy flows through producers, consumers, and decomposers",
    "Life Processes": "organisms carry out nutrition, respiration, transport, and excretion",
    "Heredity": "traits can pass from parents to offspring through genetic information",
    "Evolution": "populations change over generations through inherited variation and selection",
    "Environment": "human actions can affect natural systems and resources",
    "Algorithms": "an algorithm is an ordered method for solving a problem",
    "Data Representation": "computers encode text, images, sound, and numbers in binary forms",
    "Programming Basics": "programs use variables, input, output, decisions, and repetition",
    "Networks": "networks allow devices to communicate using agreed protocols",
    "Problem Solving": "complex problems can be decomposed into smaller testable tasks",
    "Cyber Safety": "safe online practice protects accounts, privacy, and devices",
    "Programming": "program structure and testing help turn an algorithm into reliable software",
    "Functions": "a function is a reusable named block that can accept input and return a result",
    "Lists": "a list stores an ordered collection of values that can be accessed by index",
    "File Handling": "file operations open, read, write, and close persistent data safely",
    "Debugging": "debugging identifies, isolates, and corrects program errors",
    "Error Analysis": "measurement uncertainty should be estimated and reported honestly",
    "Mechanics Lab": "controlled measurements link forces, motion, and calculated quantities",
    "Electricity Lab": "circuits should be measured systematically using suitable meters",
    "Graphing": "a graph should use labelled axes, sensible scales, and an appropriate trend",
    "Salt Analysis": "chemical tests can identify ions through characteristic observations",
    "Molarity": "molarity expresses the amount of solute per unit volume of solution",
    "Lab Calculations": "practical calculations require units, significant figures, and justified working",
    "Specimen Study": "specimens should be observed systematically and recorded accurately",
    "Enzymes": "enzymes are biological catalysts whose activity depends on conditions",
    "Experimental Design": "a fair investigation controls variables and repeats measurements",
    "Semiconductor Lab": "semiconductors have electrical behaviour that can be controlled in circuits",
    "Optics Lab": "optics practical work measures how light travels and forms images",
    "Oscilloscope": "an oscilloscope displays electrical signals as changing voltage over time",
    "Data Analysis": "experimental data should be organised, interpreted, and evaluated for uncertainty",
    "Organic Analysis": "organic substances can be identified using properties and chemical tests",
    "Electrochemical Cells": "electrochemical cells convert chemical energy into electrical energy",
    "Kinetics Lab": "rate experiments compare how controlled changes affect reaction speed",
    "Ecology Sampling": "sampling methods estimate populations and distributions in an ecosystem",
    "Slide Preparation": "a prepared slide must be thin, clean, and labelled for clear observation",
    "Python Projects": "a complete program should be planned, implemented, tested, and documented",
    "SQL Database": "a database stores linked records that can be filtered with queries",
    "Documentation": "documentation explains how software works and how it should be used",
}

SUBJECT_FOCUS = {
    "Mathematics": "the definitions, rules, and worked methods for",
    "Physics": "the quantities, units, models, and applications of",
    "Chemistry": "the particles, reactions, and evidence connected to",
    "Biology": "the structures, processes, and relationships in",
    "English": "the key language choices and interpretations in",
    "Computer Science": "the logic, data, and problem-solving techniques used in",
    "Practical": "the equipment, method, safe procedure, and results for",
}


def offline_idea(topic: str, subject: str) -> str:
    if topic in TOPIC_IDEAS:
        return TOPIC_IDEAS[topic]
    category = "Practical" if "Practical" in subject else subject
    return f"{SUBJECT_FOCUS.get(category, 'the key concepts and applications of')} {topic}"


QUESTION_SEEDS: dict[str, list[tuple[str, str, list[str], str]]] = {
    "Sets": [
        ("If A = {1, 2, 3} and B = {3, 4, 5}, what is A ∩ B?", "{3}", ["{1, 2}", "{4, 5}", "{1, 2, 3, 4, 5}"], "The intersection contains only elements present in both sets."),
        ("Which relation is required for a set to be a subset of another set?", "Every element of the first set belongs to the second set.", ["The sets have equal size only.", "The first set has no elements.", "The sets share exactly one element."], "A subset condition concerns membership of every element."),
    ],
    "Functions": [
        ("For f(x) = 2x + 1, what is f(3)?", "7", ["5", "6", "8"], "Substituting x = 3 gives 2(3) + 1 = 7."),
        ("Which relation is a function?", "Each input has exactly one output.", ["Each output has exactly one input only.", "One input has two outputs.", "Inputs are optional."], "A function assigns one output to each input."),
    ],
    "Limits": [
        ("What value does (x² − 4)/(x − 2) approach as x approaches 2?", "4", ["0", "2", "Undefined forever"], "For x not equal to 2, the expression simplifies to x + 2, which approaches 4."),
        ("A limit describes which idea?", "The value a function approaches near an input.", ["Only the exact value at the input.", "The largest output of a function.", "A value unrelated to the input."], "Limits describe nearby behaviour, not only direct substitution."),
    ],
    "Kinematics": [
        ("A runner travels 100 m in 20 s. What is the average speed?", "5 m/s", ["2 m/s", "20 m/s", "120 m/s"], "Average speed equals distance divided by time: 100 ÷ 20."),
        ("Which quantity includes direction?", "Velocity", ["Speed", "Distance", "Time"], "Velocity is displacement per time and includes direction."),
    ],
    "Dynamics": [
        ("A 2 kg object accelerates at 3 m/s². What net force acts on it?", "6 N", ["1.5 N", "5 N", "9 N"], "Newton's second law gives F = ma = 2 × 3."),
        ("If the net force on an object is zero, what can be true?", "It can move at constant velocity.", ["It must speed up.", "It must stop immediately.", "It must change direction."], "Zero net force means zero acceleration, not necessarily zero velocity."),
    ],
    "Mole Concept": [
        ("How many moles are in 18 g of water (molar mass 18 g/mol)?", "1 mol", ["0.5 mol", "18 mol", "36 mol"], "Moles equal mass divided by molar mass."),
        ("What does one mole represent?", "6.022 × 10²³ particles", ["One gram exactly", "One litre exactly", "One atom only"], "A mole is defined using Avogadro's number."),
    ],
    "Chemical Bonding": [
        ("What happens in ionic bonding?", "Electrons are transferred and oppositely charged ions attract.", ["Atoms share all their nuclei.", "Neutrons are exchanged.", "Electrons disappear."], "Ionic bonds result from electrostatic attraction between ions."),
        ("Why do atoms form chemical bonds?", "Bonding can produce a more stable arrangement.", ["To remove every electron.", "To create new elements instantly.", "To eliminate all forces."], "Bonding lowers energy or gives a stable electron configuration."),
    ],
    "Cell Division": [
        ("What is the main outcome of mitosis?", "Two genetically similar daughter cells", ["Four genetically different gametes", "One cell with half the chromosomes", "No new cells"], "Mitosis supports growth and repair by producing similar cells."),
        ("Why is DNA copied before mitosis?", "Each daughter cell needs a complete set of genetic information.", ["To change the species.", "To remove chromosomes.", "To stop cell growth."], "DNA replication ensures both cells receive genetic material."),
    ],
    "Genetics": [
        ("What is a gene?", "A DNA section containing information for a trait or product", ["A type of cell membrane", "A protein only", "A whole organism"], "Genes are units of inherited information encoded in DNA."),
        ("Which statement best describes an allele?", "An alternative version of a gene", ["A different species", "A cell organelle", "A type of chromosome pair only"], "Alleles are versions of the same gene at a locus."),
    ],
    "Object-Oriented Programming": [
        ("What is the relationship between a class and an object?", "A class is a blueprint; an object is an instance created from it.", ["An object creates every class automatically.", "They are unrelated to code.", "A class can contain no data."], "Classes describe structure and behaviour; objects are usable instances."),
        ("Which feature lets an object keep its data together with related methods?", "Encapsulation", ["Compilation", "Deletion", "Formatting"], "Encapsulation bundles state and behaviour."),
    ],
    "SQL Queries": [
        ("Which SQL command retrieves rows from a table?", "SELECT", ["DELETE", "DROP", "UPDATE"], "SELECT reads data without modifying it."),
        ("What does a WHERE clause do in a SELECT query?", "Filters rows using a condition", ["Creates a new database", "Always sorts columns", "Deletes the table"], "WHERE selects only records that satisfy a condition."),
    ],
    "Titration": [
        ("What indicates the end point in a titration with an indicator?", "A persistent colour change", ["The solution boiling", "The burette becoming empty", "The flask breaking"], "A persistent colour change signals the required reaction point."),
        ("Why should a burette reading be taken at eye level?", "To reduce parallax error", ["To increase the volume", "To change concentration", "To warm the solution"], "Eye-level readings prevent apparent shifts in the meniscus."),
    ],
    "Microscopy": [
        ("Why is the low-power objective normally used first?", "It gives a wider field of view to locate the specimen.", ["It always gives the highest magnification.", "It eliminates the need for focus.", "It changes the specimen colour."], "Starting low makes focusing and locating the specimen safer."),
        ("What should be included with a biological drawing?", "Clear labels and an appropriate scale or magnification", ["Decorative shading only", "Unlabelled colours", "No observations"], "Scientific drawings communicate observed structures precisely."),
    ],
}


def _local_variants(topic: str, subject: str, idea: str) -> list[tuple[str, str, list[str], str]]:
    """Produce many original, topic-specific application and reasoning prompts."""
    return [
        (f"Which statement is most accurate about {topic} in {subject}?", idea.capitalize() + ".", [f"{topic} has no definitions or evidence.", f"{topic} is unrelated to all other {subject} topics.", f"{topic} only needs one memorised answer."], f"The central idea is that {idea}."),
        (f"A student is revising {topic}. Which step best checks understanding?", f"Explain how {idea}, then apply it to a new example.", ["Copy notes without reading them.", "Choose an answer before examining the question.", "Use a rule from an unrelated subject."], f"Application and explanation demonstrate understanding of {topic}."),
        (f"Which type of evidence is most useful when checking a {topic} answer?", f"Working that is consistent with the idea that {idea}.", ["A guess without a method.", "An answer with no units or conditions checked.", "A result copied from an unrelated question."], f"Checking the method against the topic idea helps identify errors."),
        (f"Which misconception should a learner avoid in {topic}?", f"Ignoring that {idea}.", ["Using a relevant definition.", "Checking units and conditions.", "Testing an answer with an example."], f"Strong revision begins by protecting the central concept of {topic}."),
        (f"How can {topic} be connected to a real problem-solving task?", f"Use the principle that {idea} to interpret the given information.", ["Ignore all available data.", "Assume every answer is correct without checking.", "Replace the topic with an unrelated one."], f"Real application requires matching information to the governing concept."),
        (f"When comparing two methods for a {topic} problem, what should be prioritised?", f"The method that correctly uses the fact that {idea}.", ["The shortest-looking answer without reasoning.", "The method with the most copied words.", "A method that ignores the question conditions."], f"A good method is justified by the principle behind the topic."),
        (f"What is a reliable self-test for {topic}?", f"Solve a fresh example and explain why {idea}.", ["Read the title only.", "Memorise the order of answer choices.", "Avoid checking any mistakes."], f"Fresh examples reveal whether the concept is understood rather than remembered."),
        (f"A result in {topic} seems surprising. What should be checked first?", f"Whether the result still fits the idea that {idea}.", ["Whether the handwriting looks neat.", "Whether another subject has a similar word.", "Whether the first guess can be kept unchanged."], f"A concept check is more useful than changing an answer blindly."),
        (f"Which classroom discussion question is most relevant to {topic}?", f"Why does it matter that {idea}?", ["Why should definitions be ignored?", "Why should all examples be skipped?", "Why does evidence never matter?"], f"Asking why a principle works deepens conceptual understanding."),
        (f"Which revision plan is most effective for {topic}?", f"Review the key idea, practise a worked example, then correct errors using {idea}.", ["Study unrelated content only.", "Skip feedback after an incorrect answer.", "Memorise punctuation instead of the concept."], f"Effective revision connects concept review, practice, and feedback."),
    ]


def fallback_questions(data: QuizCreateIn) -> list[dict]:
    """Create a scalable bank of original topic-specific questions without copying textbook exercises."""
    questions: list[dict] = []
    starting_index = DIFFICULTIES.index(data.starting_difficulty)
    topic_usage: dict[str, int] = {}
    for index in range(data.question_count):
        topic = data.topics[index % len(data.topics)]
        level = DIFFICULTIES[(starting_index + index) % len(DIFFICULTIES)]
        idea = offline_idea(topic, data.subject)
        pool = QUESTION_SEEDS.get(topic, []) + _local_variants(topic, data.subject, idea)
        use_index = topic_usage.get(topic, 0)
        prompt, correct, distractors, explanation = pool[use_index % len(pool)]
        topic_usage[topic] = use_index + 1
        options = [correct, *distractors]
        correct_index = (index + use_index) % 4
        options = options[-correct_index:] + options[:-correct_index] if correct_index else options
        questions.append({"topic": topic, "prompt": prompt, "options": options, "correct_index": correct_index, "explanation": explanation, "difficulty": level})
    return questions


async def ai_questions(data: QuizCreateIn) -> list[dict]:
    """Ask the configured AI provider for structured questions; fall back safely on any failure."""
    endpoint = os.getenv("BUILT_IN_FORGE_API_URL")
    api_key = os.getenv("BUILT_IN_FORGE_API_KEY")
    if not endpoint or not api_key:
        return []
    prompt = (
        "Return JSON only, using the form {\"questions\":[...]}. Create "
        f"{data.question_count} accurate grade {data.grade} {data.subject} multiple-choice questions about "
        f"{', '.join(data.topics)}. Every question needs topic, prompt, options (exactly four strings), "
        "correct_index (0-3), explanation, and difficulty. Difficulty must be exactly easy, medium, or hard. "
        f"The first question must be {data.starting_difficulty}."
    )
    try:
        async with httpx.AsyncClient(timeout=18) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 2200},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            generated = parsed.get("questions", [])
            valid: list[dict] = []
            for item in generated:
                if (
                    isinstance(item, dict)
                    and item.get("difficulty") in DIFFICULTIES
                    and isinstance(item.get("options"), list)
                    and len(item["options"]) == 4
                    and isinstance(item.get("correct_index"), int)
                    and 0 <= item["correct_index"] <= 3
                ):
                    valid.append(item)
            return valid[: data.question_count] if len(valid) >= data.question_count else []
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return []


async def make_questions(data: QuizCreateIn) -> list[dict]:
    return await ai_questions(data) or fallback_questions(data)


def question_out(question: Question | None) -> QuestionOut | None:
    if question is None:
        return None
    return QuestionOut(
        id=question.id,
        topic=question.topic,
        prompt=question.prompt,
        options=question.options,
        difficulty=question.difficulty,  # type: ignore[arg-type]
    )


def active_question(db: Session, quiz_id: int) -> Question | None:
    return db.scalar(
        select(Question)
        .where(Question.quiz_id == quiz_id, Question.delivered.is_(True))
        .where(~Question.id.in_(select(Answer.question_id).where(Answer.quiz_id == quiz_id)))
        .order_by(Question.id)
        .limit(1)
    )


def serve_next_question(db: Session, quiz: Quiz) -> Question | None:
    preferred = db.scalar(
        select(Question)
        .where(Question.quiz_id == quiz.id, Question.delivered.is_(False), Question.difficulty == quiz.adaptive_difficulty)
        .order_by(Question.id)
        .limit(1)
    )
    question = preferred or db.scalar(
        select(Question).where(Question.quiz_id == quiz.id, Question.delivered.is_(False)).order_by(Question.id).limit(1)
    )
    if question:
        question.delivered = True
    return question


def as_quiz_out(db: Session, quiz: Quiz) -> QuizOut:
    return QuizOut(
        id=quiz.id,
        grade=quiz.grade,
        subject=quiz.subject,
        topics=quiz.topics,
        mode=quiz.mode,
        timer_minutes=quiz.timer_minutes,
        expires_at=quiz.expires_at,
        adaptive_difficulty=quiz.adaptive_difficulty,  # type: ignore[arg-type]
        status=quiz.status,
        score_percent=quiz.score_percent,
        current_question=question_out(active_question(db, quiz.id)),
    )


def owned_quiz(db: Session, quiz_id: int, user_id: int) -> Quiz:
    quiz = db.get(Quiz, quiz_id)
    if quiz is None or quiz.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found.")
    return quiz


def ensure_open(quiz: Quiz) -> None:
    if quiz.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This quiz is already complete.")
    if quiz.expires_at and utc_now() > quiz.expires_at:
        quiz.status = "expired"
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Your student-configured timer has ended.")


def adapt(db: Session, quiz: Quiz, correct: bool) -> Difficulty:
    recent = db.scalars(select(Answer).where(Answer.quiz_id == quiz.id).order_by(Answer.id.desc()).limit(2)).all()
    index = DIFFICULTIES.index(quiz.adaptive_difficulty)  # type: ignore[arg-type]
    if len(recent) == 2 and all(item.is_correct for item in recent):
        index = min(index + 1, 2)
    elif len(recent) == 2 and not any(item.is_correct for item in recent):
        index = max(index - 1, 0)
    quiz.adaptive_difficulty = DIFFICULTIES[index]
    return DIFFICULTIES[index]


def topic_insights(db: Session, user_id: int) -> list[dict]:
    answers = db.execute(
        select(Question.topic, Answer.is_correct)
        .join(Answer, Answer.question_id == Question.id)
        .where(Answer.user_id == user_id)
    ).all()
    grouped: dict[str, list[bool]] = {}
    for topic, correct in answers:
        grouped.setdefault(topic, []).append(correct)
    result = []
    for topic, values in grouped.items():
        percent = round(100 * sum(values) / len(values), 1)
        label = "weak" if percent < 50 else "average" if percent < 80 else "strong"
        result.append({"topic": topic, "accuracy": percent, "category": label, "attempts": len(values)})
    return sorted(result, key=lambda item: item["accuracy"])


def complete_quiz(db: Session, quiz: Quiz) -> dict:
    attempts = db.scalars(select(Answer).where(Answer.quiz_id == quiz.id)).all()
    questions = db.scalars(select(Question).where(Question.quiz_id == quiz.id)).all()
    total = len(questions)
    correct = sum(1 for answer in attempts if answer.is_correct)
    score = round((correct / total) * 100, 1) if total else 0.0
    quiz.score_percent = score
    quiz.status = "completed"
    quiz.completed_at = utc_now()
    return {
        "quiz_id": quiz.id,
        "score_percent": score,
        "correct_answers": correct,
        "total_questions": total,
        "topic_insights": topic_insights(db, quiz.user_id),
    }


# ---------------------------------------------------------------------------
# FastAPI application and REST routes
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="SmartQuiz AI — Single File FastAPI Project",
    version="1.0.0",
    description="JWT-secured adaptive quiz API for grades 9–12, contained in one main.py file.",
    lifespan=lifespan,
)


@app.get("/api/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "application": "SmartQuiz AI", "single_file": True}


@app.post("/api/auth/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED, tags=["authentication"])
def register(payload: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    email = str(payload.email).lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account already exists for this email.")
    user = User(name=payload.name.strip(), email=email, password_hash=hash_password(payload.password), grade=payload.grade)
    db.add(user)
    db.flush()
    db.add(Reminder(user_id=user.id))
    db.commit()
    return TokenOut(access_token=create_token(user.id))


@app.post("/api/auth/login", response_model=TokenOut, tags=["authentication"])
def login(payload: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.email == str(payload.email).lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    return TokenOut(access_token=create_token(user.id))


@app.get("/api/users/me", response_model=UserOut, tags=["private"])
def profile(user: User = Depends(current_user)) -> UserOut:
    return UserOut(id=user.id, name=user.name, email=user.email, grade=user.grade)


@app.get("/api/catalog", tags=["question-bank"])
def catalog() -> dict[int, dict[str, list[str]]]:
    return CATALOG


@app.post("/api/quizzes", response_model=QuizOut, status_code=status.HTTP_201_CREATED, tags=["quizzes"])
async def create_quiz(payload: QuizCreateIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> QuizOut:
    questions = await make_questions(payload)
    now = utc_now()
    quiz = Quiz(
        user_id=user.id,
        grade=payload.grade,
        subject=payload.subject.strip(),
        topics=[topic.strip() for topic in payload.topics],
        timer_minutes=payload.timer_minutes,
        expires_at=now + timedelta(minutes=payload.timer_minutes) if payload.timer_minutes else None,
        adaptive_difficulty=payload.starting_difficulty,
    )
    db.add(quiz)
    db.flush()
    for item in questions:
        db.add(Question(quiz_id=quiz.id, **item))
    db.flush()
    serve_next_question(db, quiz)
    db.commit()
    return as_quiz_out(db, quiz)


@app.get("/api/quizzes", response_model=list[QuizOut], tags=["quizzes"])
def list_quizzes(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[QuizOut]:
    quizzes = db.scalars(select(Quiz).where(Quiz.user_id == user.id).order_by(Quiz.started_at.desc())).all()
    return [as_quiz_out(db, quiz) for quiz in quizzes]


@app.get("/api/quizzes/{quiz_id}", response_model=QuizOut, tags=["quizzes"])
def get_quiz(quiz_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> QuizOut:
    return as_quiz_out(db, owned_quiz(db, quiz_id, user.id))


@app.post("/api/quizzes/{quiz_id}/answers", response_model=AnswerOut, tags=["quizzes"])
def answer_quiz(quiz_id: int, payload: AnswerIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> AnswerOut:
    quiz = owned_quiz(db, quiz_id, user.id)
    ensure_open(quiz)
    question = db.get(Question, payload.question_id)
    if question is None or question.quiz_id != quiz.id or not question.delivered:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The active question was not found.")
    if db.scalar(select(Answer).where(Answer.question_id == question.id)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This question was already answered.")
    correct = payload.selected_index == question.correct_index
    db.add(Answer(quiz_id=quiz.id, question_id=question.id, user_id=user.id, selected_index=payload.selected_index, is_correct=correct))
    db.flush()
    difficulty = adapt(db, quiz, correct)
    next_item = serve_next_question(db, quiz)
    finished = next_item is None
    if finished:
        complete_quiz(db, quiz)
    db.commit()
    return AnswerOut(
        is_correct=correct,
        explanation=question.explanation,
        adaptive_difficulty=difficulty,
        next_question=question_out(next_item),
        completed=finished,
    )


@app.post("/api/quizzes/{quiz_id}/complete", tags=["quizzes"])
def finish_quiz(quiz_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    quiz = owned_quiz(db, quiz_id, user.id)
    if quiz.status == "completed":
        return {"quiz_id": quiz.id, "score_percent": quiz.score_percent, "topic_insights": topic_insights(db, user.id)}
    result = complete_quiz(db, quiz)
    db.commit()
    return result


@app.delete("/api/quizzes/{quiz_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["quizzes"])
def delete_quiz(quiz_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    quiz = owned_quiz(db, quiz_id, user.id)
    question_ids = db.scalars(select(Question.id).where(Question.quiz_id == quiz.id)).all()
    if question_ids:
        db.query(Answer).filter(Answer.question_id.in_(question_ids)).delete(synchronize_session=False)
    db.query(Question).filter(Question.quiz_id == quiz.id).delete(synchronize_session=False)
    db.delete(quiz)
    db.commit()


@app.post("/api/quizzes/revision", response_model=QuizOut, status_code=status.HTTP_201_CREATED, tags=["revision"])
def revision_quiz(user: User = Depends(current_user), db: Session = Depends(get_db)) -> QuizOut:
    incorrect = db.execute(
        select(Question, Quiz)
        .join(Answer, Answer.question_id == Question.id)
        .join(Quiz, Quiz.id == Question.quiz_id)
        .where(Answer.user_id == user.id, Answer.is_correct.is_(False))
        .order_by(Answer.answered_at.desc())
    ).all()
    selected: list[tuple[Question, Quiz]] = []
    used: set[int] = set()
    for question, original_quiz in incorrect:
        key = question.source_question_id or question.id
        if key not in used:
            selected.append((question, original_quiz))
            used.add(key)
    if not selected:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No previously incorrect answers are available for revision.")
    first_question, first_quiz = selected[0]
    quiz = Quiz(
        user_id=user.id,
        grade=first_quiz.grade,
        subject=first_quiz.subject,
        topics=list(dict.fromkeys(question.topic for question, _ in selected)),
        mode="revision",
        adaptive_difficulty="medium",
    )
    db.add(quiz)
    db.flush()
    for question, _ in selected[:15]:
        db.add(
            Question(
                quiz_id=quiz.id,
                source_question_id=question.source_question_id or question.id,
                topic=question.topic,
                prompt=question.prompt,
                options=question.options,
                correct_index=question.correct_index,
                explanation=question.explanation,
                difficulty=question.difficulty,
            )
        )
    db.flush()
    serve_next_question(db, quiz)
    db.commit()
    return as_quiz_out(db, quiz)


@app.get("/api/reports/daily", tags=["reports"])
def report(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    completed = db.scalars(select(Quiz).where(Quiz.user_id == user.id, Quiz.status == "completed").order_by(Quiz.completed_at)).all()
    scores = [quiz.score_percent for quiz in completed if quiz.score_percent is not None]
    average = round(sum(scores) / len(scores), 1) if scores else None
    improvement = round(scores[-1] - scores[-2], 1) if len(scores) >= 2 else None
    days = []
    for offset in range(6, -1, -1):
        day = (utc_now() - timedelta(days=offset)).date().isoformat()
        day_scores = [score for quiz in completed if quiz.completed_at and quiz.completed_at.date().isoformat() == day for score in [quiz.score_percent] if score is not None]
        days.append({"date": day, "average_score": round(sum(day_scores) / len(day_scores), 1) if day_scores else None})
    return {
        "completed_quizzes": len(completed),
        "average_score": average,
        "improvement": improvement,
        "trend": days,
        "topic_breakdown": topic_insights(db, user.id),
    }


@app.get("/api/reminders/today", response_model=ReminderOut, tags=["reminders"])
def today_reminder(user: User = Depends(current_user), db: Session = Depends(get_db)) -> ReminderOut:
    reminder = db.scalar(select(Reminder).where(Reminder.user_id == user.id))
    if reminder is None:
        reminder = Reminder(user_id=user.id)
        db.add(reminder)
        db.commit()
    try:
        local = datetime.now(ZoneInfo(reminder.timezone))
    except ZoneInfoNotFoundError:
        local = datetime.now(ZoneInfo("UTC"))
    today = local.date().isoformat()
    show = reminder.enabled and reminder.last_dismissed_on != today and local.strftime("%H:%M") >= reminder.reminder_time
    return ReminderOut(
        enabled=reminder.enabled,
        reminder_time=reminder.reminder_time,
        timezone=reminder.timezone,
        show_today=show,
        message="Your daily quiz is ready. A focused session keeps your progress moving." if show else None,
    )


@app.put("/api/reminders", response_model=ReminderOut, tags=["reminders"])
def update_reminder(payload: ReminderIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> ReminderOut:
    try:
        ZoneInfo(payload.timezone)
    except ZoneInfoNotFoundError:
        raise HTTPException(status_code=422, detail="Use a valid IANA timezone, for example UTC or Asia/Karachi.")
    reminder = db.scalar(select(Reminder).where(Reminder.user_id == user.id)) or Reminder(user_id=user.id)
    reminder.enabled = payload.enabled
    reminder.reminder_time = payload.reminder_time
    reminder.timezone = payload.timezone
    db.add(reminder)
    db.commit()
    return ReminderOut(**payload.model_dump(), show_today=False, message=None)


@app.post("/api/reminders/dismiss", status_code=status.HTTP_204_NO_CONTENT, tags=["reminders"])
def dismiss_reminder(user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    reminder = db.scalar(select(Reminder).where(Reminder.user_id == user.id)) or Reminder(user_id=user.id)
    try:
        reminder.last_dismissed_on = datetime.now(ZoneInfo(reminder.timezone)).date().isoformat()
    except ZoneInfoNotFoundError:
        reminder.last_dismissed_on = datetime.now(ZoneInfo("UTC")).date().isoformat()
    db.add(reminder)
    db.commit()


# ---------------------------------------------------------------------------
# Embedded student interface (no templates or separate static files required)
# ---------------------------------------------------------------------------

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartQuiz AI</title><style>
:root{--ink:#071824;--panel:#0b2530;--cream:#f7f6f2;--teal:#09857c;--mint:#93e1d5;--coral:#ed9178;--muted:#73818a}*{box-sizing:border-box}body{margin:0;font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;color:var(--ink);background:var(--cream)}button,input,select{font:inherit}button{cursor:pointer;border:0}.hidden{display:none!important}.auth{min-height:100vh;display:grid;grid-template-columns:1.2fr .8fr;background:radial-gradient(circle at 55% 30%,#0c4950 0,var(--ink) 55%);color:#fff}.brand{padding:48px max(8vw,32px);display:flex;flex-direction:column;justify-content:center}.logo{font-weight:800;font-size:19px}.mark{display:inline-grid;place-items:center;width:29px;height:29px;border-radius:9px;background:var(--coral);margin-right:8px;color:var(--ink)}.eyebrow{letter-spacing:.16em;text-transform:uppercase;color:var(--mint);font-size:11px;margin:80px 0 16px}.hero{font:clamp(48px,6vw,86px)/.95 Georgia,serif;margin:0;max-width:680px}.hero em{color:var(--mint)}.sub{max-width:500px;font-size:18px;color:#c4d1d2;margin-top:26px}.stats{display:flex;gap:28px;margin-top:40px;color:#b8c7c8}.stats b{color:#fff;margin-right:6px}.card{align-self:center;background:#fff;color:var(--ink);border-radius:24px;padding:34px;width:min(420px,calc(100% - 36px));margin:18px;box-shadow:0 26px 80px #00101855}.tabs{display:flex;gap:26px;border-bottom:1px solid #dce2e1;margin-bottom:22px}.tabs button{background:none;padding:0 0 14px;font-weight:700;color:var(--muted)}.tabs button.active{color:var(--ink);border-bottom:3px solid var(--teal)}label{display:block;font-size:13px;font-weight:700;margin:14px 0 5px}input,select{width:100%;border:1px solid #d6dedc;border-radius:10px;padding:12px;background:#fff}.primary{background:var(--teal);color:white;width:100%;padding:14px;border-radius:10px;font-weight:800;margin-top:18px;box-shadow:0 8px 18px #09857c33}.error{color:#b42424;font-size:13px;margin-top:10px}.app{min-height:100vh;display:grid;grid-template-columns:230px 1fr;background:#f5f6f3}.side{background:#061824;color:#b9c9cc;padding:26px 16px;display:flex;flex-direction:column}.side .logo{color:#fff;margin-bottom:48px}.nav{display:grid;gap:7px}.nav button{background:transparent;color:inherit;text-align:left;padding:12px 14px;border-radius:9px}.nav button.active,.nav button:hover{background:#12323e;color:#fff}.profile{margin-top:auto;padding:13px;background:#102d38;border-radius:12px;color:white}.main{padding:40px;max-width:1200px;width:100%;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:34px}.top h1{font:38px/1 Georgia,serif;margin:0}.badge{background:#e0f2ed;color:#087269;border-radius:7px;padding:5px 8px;font-size:12px;font-weight:800}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.metric,.section{background:#fff;border:1px solid #e5e9e6;border-radius:18px;padding:20px;box-shadow:0 5px 18px #0b253008}.metric strong{font-size:32px;display:block}.section{margin-top:18px}.section h2{font:25px Georgia,serif;margin:0 0 10px}.formgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.topics{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.topic{border:1px solid #cfd9d7;border-radius:20px;padding:7px 11px;background:white}.topic.selected{background:#d8f1eb;color:#056e66;border-color:#75c7bb}.row{display:flex;justify-content:space-between;align-items:center;gap:12px}.small{width:auto;padding:9px 13px;margin:0}.quiz{max-width:820px;margin:8vh auto}.question{font:clamp(30px,4vw,52px)/1.06 Georgia,serif;margin:25px 0}.options{display:grid;gap:10px}.option{padding:16px;border:1px solid #d6e0dd;border-radius:12px;background:white;text-align:left}.option:hover{border-color:var(--teal);background:#f0fbf8}.result{padding:11px;border-radius:9px;margin-top:16px}.good{background:#ddf5e9;color:#0b6753}.bad{background:#fff0ec;color:#b94a36}.insight{display:flex;justify-content:space-between;border-top:1px solid #e4e8e5;padding:10px 0}.weak{color:#b94a36}.average{color:#be7a09}.strong{color:#087269}.reminder{background:#ffe9ba;padding:14px;border-radius:12px;margin:0 0 20px}.logout{background:#fff;border:1px solid #d6dfdc;color:#37474d;padding:9px 12px;border-radius:9px}@media(max-width:800px){.auth{grid-template-columns:1fr}.brand{padding:36px}.hero{font-size:54px}.app{grid-template-columns:1fr}.side{display:none}.main{padding:22px}.grid,.formgrid{grid-template-columns:1fr}}
</style></head><body>
<section id="auth" class="auth"><div class="brand"><div class="logo"><span class="mark">S</span>SmartQuiz AI</div><p class="eyebrow">A calmer way to get better</p><h1 class="hero">Study with a plan that <em>adapts</em> to you.</h1><p class="sub">Personalized quizzes for grades 9–12 that respond to your answers and make every study session count.</p><p class="stats"><span><b>9–12</b> Grades</span><span><b>3</b> Difficulty levels</span><span><b>1</b> Focused path</span></p></div><div class="card"><div class="tabs"><button id="loginTab" class="active">Welcome back</button><button id="registerTab">Create account</button></div><form id="loginForm"><label>Email</label><input name="email" type="email" placeholder="you@example.com" required><label>Password</label><input name="password" type="password" required><button class="primary">Sign in →</button></form><form id="registerForm" class="hidden"><label>Full name</label><input name="name" required><label>Email</label><input name="email" type="email" required><label>Password</label><input name="password" type="password" minlength="8" required><label>Grade</label><select name="grade"><option value="9">Grade 9</option><option value="10">Grade 10</option><option value="11">Grade 11</option><option value="12">Grade 12</option></select><button class="primary">Create my account →</button></form><p id="authError" class="error"></p></div></section>
<section id="app" class="app hidden"><aside class="side"><div class="logo"><span class="mark">S</span>SmartQuiz AI</div><nav class="nav"><button data-view="overview" class="active">◈ Overview</button><button data-view="builder">＋ New quiz</button><button id="revision">↻ Revision</button><button data-view="reports">⌁ Reports</button></nav><div id="profile" class="profile"></div></aside><main class="main"><div class="top"><span>Your learning space</span><button id="logout" class="logout">Sign out</button></div><div id="overview" class="view"></div><div id="builder" class="view hidden"></div><div id="reports" class="view hidden"></div><div id="runner" class="view hidden"></div></main></section>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)]; let token=localStorage.smartquizToken, catalog={}, user=null, activeQuiz=null;
const api=async(path,opt={})=>{let r=await fetch(path,{...opt,headers:{'Content-Type':'application/json',...(token?{Authorization:'Bearer '+token}:{}),...(opt.headers||{})}});let d=r.status===204?null:await r.json();if(!r.ok)throw Error(Array.isArray(d.detail)?d.detail.map(x=>x.msg).join(' '):d.detail||'Request failed');return d};
function show(id){$$('.view').forEach(x=>x.classList.add('hidden'));$('#'+id).classList.remove('hidden');$$('.nav button[data-view]').forEach(x=>x.classList.toggle('active',x.dataset.view===id))}
function authMode(register){$('#loginForm').classList.toggle('hidden',register);$('#registerForm').classList.toggle('hidden',!register);$('#loginTab').classList.toggle('active',!register);$('#registerTab').classList.toggle('active',register);$('#authError').textContent=''}
$('#loginTab').onclick=()=>authMode(false);$('#registerTab').onclick=()=>authMode(true);
async function authenticate(e,register){e.preventDefault();try{let f=Object.fromEntries(new FormData(e.target));if(register)f.grade=+f.grade;let d=await api('/api/auth/'+(register?'register':'login'),{method:'POST',body:JSON.stringify(f)});token=d.access_token;localStorage.smartquizToken=token;await boot()}catch(err){$('#authError').textContent=err.message}}
$('#loginForm').onsubmit=e=>authenticate(e,false);$('#registerForm').onsubmit=e=>authenticate(e,true);$('#logout').onclick=()=>{localStorage.removeItem('smartquizToken');location.reload()};
async function boot(){user=await api('/api/users/me');catalog=await api('/api/catalog');$('#auth').classList.add('hidden');$('#app').classList.remove('hidden');$('#profile').innerHTML='<b>'+user.name+'</b><br><small>Grade '+user.grade+'</small>';await dashboard();show('overview')}
async function dashboard(){let r=await api('/api/reports/daily'), rem=await api('/api/reminders/today');$('#overview').innerHTML=(rem.show_today?'<div class="reminder"><b>DAILY NUDGE</b><br>'+rem.message+' <button id="dismiss" class="small">Not now</button></div>':'')+'<h1>Good morning, <em>'+user.name.split(' ')[0]+'</em>.</h1><div class="grid"><div class="metric"><span>Completed quizzes</span><strong>'+r.completed_quizzes+'</strong></div><div class="metric"><span>Average score</span><strong>'+(r.average_score??'—')+(r.average_score!==null?'%':'')+'</strong></div><div class="metric"><span>Improvement</span><strong>'+(r.improvement===null?'—':(r.improvement>0?'+':'')+r.improvement+'%')+'</strong></div></div><section class="section"><div class="row"><h2>Topic pulse</h2><button id="reminderSettings" class="logout">Daily reminder</button></div>'+insights(r.topic_breakdown)+'</section>';if($('#dismiss'))$('#dismiss').onclick=async()=>{await api('/api/reminders/dismiss',{method:'POST'});dashboard()};$('#reminderSettings').onclick=reminderSettings}
function insights(items){return items.length?items.map(x=>'<div class="insight"><span>'+x.topic+'</span><span class="'+x.category+'">'+x.category+' · '+x.accuracy+'%</span></div>').join(''):'<p>Finish a quiz to reveal your topic pulse.</p>'}
function subjects(){return Object.keys(catalog[user.grade]||{})}
function renderBuilder(){let subs=subjects();$('#builder').innerHTML='<h1>Make a quiz that <em>meets you</em> where you are.</h1><section class="section"><form id="quizForm"><div class="formgrid"><div><label>Grade</label><select id="grade">'+[9,10,11,12].map(g=>'<option '+(g===user.grade?'selected':'')+' value="'+g+'">Grade '+g+'</option>').join('')+'</select></div><div><label>Subject</label><select id="subject"></select></div><div><label>Starting difficulty</label><select id="difficulty"><option>easy</option><option selected>medium</option><option>hard</option></select></div><div><label>Questions</label><select id="count"><option value="5">5 questions</option><option value="8">8 questions</option><option value="10">10 questions</option></select></div><div><label>Timed challenge</label><select id="timer"><option value="">No timer — study at your pace</option><option value="5">5 minutes</option><option value="10">10 minutes</option><option value="20">20 minutes</option></select></div></div><label>Topics</label><div id="topics" class="topics"></div><button class="primary">Generate my quiz →</button></form></section>';function sync(){let g=$('#grade').value, ss=Object.keys(catalog[g]||{});$('#subject').innerHTML=ss.map(s=>'<option>'+s+'</option>').join('');topicOptions()}function topicOptions(){let a=catalog[$('#grade').value][$('#subject').value]||[];$('#topics').innerHTML=a.map((t,i)=>'<button type="button" class="topic '+(i===0?'selected':'')+'" data-topic="'+t+'">'+t+'</button>').join('');$$('.topic').forEach(b=>b.onclick=()=>b.classList.toggle('selected'))}$('#grade').onchange=sync;$('#subject').onchange=topicOptions;sync();$('#quizForm').onsubmit=async e=>{e.preventDefault();let topics=$$('.topic.selected').map(x=>x.dataset.topic);if(!topics.length)return alert('Choose at least one topic.');try{activeQuiz=await api('/api/quizzes',{method:'POST',body:JSON.stringify({grade:+$('#grade').value,subject:$('#subject').value,topics,question_count:+$('#count').value,starting_difficulty:$('#difficulty').value,timer_minutes:$('#timer').value?+$('#timer').value:null})});renderRunner()}catch(err){alert(err.message)}}}
function renderRunner(){show('runner');let q=activeQuiz.current_question;if(!q){dashboard();show('reports');return}$('#runner').innerHTML='<div class="quiz"><div class="row"><span class="badge">'+q.topic+' · '+q.difficulty+'</span><span>'+ (activeQuiz.timer_minutes?'Timed challenge: '+activeQuiz.timer_minutes+' minutes':'Study at your pace')+'</span></div><p class="eyebrow" style="margin:32px 0 0;color:#087269">'+activeQuiz.subject+' · Grade '+activeQuiz.grade+'</p><h1 class="question">'+q.prompt+'</h1><div class="options">'+q.options.map((o,i)=>'<button class="option" data-i="'+i+'"><b>'+String.fromCharCode(65+i)+'</b> &nbsp;'+o+'</button>').join('')+'</div><div id="feedback"></div></div>';$$('.option').forEach(b=>b.onclick=()=>submitAnswer(+b.dataset.i))}
async function submitAnswer(index){try{let f=await api('/api/quizzes/'+activeQuiz.id+'/answers',{method:'POST',body:JSON.stringify({question_id:activeQuiz.current_question.id,selected_index:index})});$('#feedback').innerHTML='<div class="result '+(f.is_correct?'good':'bad')+'"><b>'+(f.is_correct?'Correct.':'Not quite.')+'</b> '+f.explanation+'<button id="continue" class="small">'+(f.completed?'See report →':'Continue →')+'</button></div>';$('#continue').onclick=async()=>{if(f.completed){await reports();show('reports')}else{activeQuiz.adaptive_difficulty=f.adaptive_difficulty;activeQuiz.current_question=f.next_question;renderRunner()}}}catch(err){alert(err.message)}}
async function reports(){let r=await api('/api/reports/daily');$('#reports').innerHTML='<h1>Your progress, <em>clearly</em>.</h1><section class="section"><h2>Daily quiz report</h2><div class="grid"><div class="metric"><span>Average score</span><strong>'+(r.average_score??'—')+(r.average_score!==null?'%':'')+'</strong></div><div class="metric"><span>Improvement</span><strong>'+(r.improvement??'—')+(r.improvement!==null?'%':'')+'</strong></div><div class="metric"><span>Quizzes completed</span><strong>'+r.completed_quizzes+'</strong></div></div></section><section class="section"><h2>Topic breakdown</h2>'+insights(r.topic_breakdown)+'</section>'}
async function reminderSettings(){let r=await api('/api/reminders/today');let time=prompt('Daily reminder time (HH:MM)',r.reminder_time);if(time===null)return;let zone=prompt('Your IANA timezone',r.timezone);if(zone===null)return;try{await api('/api/reminders',{method:'PUT',body:JSON.stringify({enabled:true,reminder_time:time,timezone:zone})});alert('Reminder saved.')}catch(err){alert(err.message)}}
$$('.nav button[data-view]').forEach(b=>b.onclick=async()=>{if(b.dataset.view==='builder')renderBuilder();if(b.dataset.view==='reports')await reports();show(b.dataset.view)});$('#revision').onclick=async()=>{try{activeQuiz=await api('/api/quizzes/revision',{method:'POST'});renderRunner()}catch(err){alert(err.message)}};
if(token)boot().catch(()=>{localStorage.removeItem('smartquizToken');token=null});
// Enhanced Overview: it stays useful even before the student has completed a quiz.
dashboard = async function(){
  let r={completed_quizzes:0,average_score:null,improvement:null,topic_breakdown:[]}, rem={show_today:false}, quizzes=[];
  try { [r,rem,quizzes]=await Promise.all([api('/api/reports/daily'),api('/api/reminders/today'),api('/api/quizzes')]); }
  catch(error){ console.warn('Overview data could not be refreshed:',error); }
  const recommended=(catalog[user.grade]?.Mathematics||Object.values(catalog[user.grade]||{})[0]||[]).slice(0,3);
  const active=quizzes.find(q=>q.status==='active'&&q.current_question);
  const firstUse=r.completed_quizzes===0;
  $('#overview').innerHTML=(rem.show_today?'<div class="reminder"><b>DAILY NUDGE</b><br>'+rem.message+' <button id="dismiss" class="small">Not now</button></div>':'')+
    '<h1>Good morning, <em>'+user.name.split(' ')[0]+'</em>.</h1>'+
    '<p>'+ (firstUse?'Start with a short personalised quiz. Your scores, improvement, and topic strengths will appear here after your first session.':'Here is your learning progress so far. Keep practising to strengthen your topic pulse.') +'</p>'+
    '<div class="grid"><div class="metric"><span>Completed quizzes</span><strong>'+r.completed_quizzes+'</strong></div><div class="metric"><span>Average score</span><strong>'+(r.average_score??'—')+(r.average_score!==null?'%':'')+'</strong></div><div class="metric"><span>Improvement</span><strong>'+(r.improvement===null?'—':(r.improvement>0?'+':'')+r.improvement+'%')+'</strong></div></div>'+
    '<section class="section"><div class="row"><h2>'+ (active?'Continue your quiz':'Your next study step') +'</h2><button id="reminderSettings" class="logout">Daily reminder</button></div>'+
    (active?'<p>Your '+active.subject+' quiz is waiting at '+active.current_question.topic+'.</p><button id="resumeQuiz" class="primary">Resume quiz →</button>':'<p>Suggested starting topics for Grade '+user.grade+': <b>'+recommended.join(' · ')+'</b></p><button id="quickStart" class="primary">Create my first quiz →</button>')+
    '</section><section class="section"><h2>Topic pulse</h2>'+insights(r.topic_breakdown)+'</section>';
  if($('#dismiss'))$('#dismiss').onclick=async()=>{await api('/api/reminders/dismiss',{method:'POST'});dashboard()};
  if($('#reminderSettings'))$('#reminderSettings').onclick=reminderSettings;
  if($('#quickStart'))$('#quickStart').onclick=()=>{renderBuilder();show('builder')};
  if($('#resumeQuiz'))$('#resumeQuiz').onclick=()=>{activeQuiz=active;renderRunner()};
};
function insights(items){return items.length?items.map(x=>'<div class="insight"><span>'+x.topic+'</span><span class="'+x.category+'">'+x.category+' · '+x.accuracy+'%</span></div>').join(''):'<p>No completed quizzes yet. Complete a short quiz to identify topics that are <b>weak</b>, <b>average</b>, or <b>strong</b>.</p>'}
// Expanded builder: students may enter their own countdown in minutes.
renderBuilder = function(){
  $('#builder').innerHTML='<h1>Make a quiz that <em>meets you</em> where you are.</h1><section class="section"><form id="quizForm"><div class="formgrid"><div><label>Grade</label><select id="grade">'+[9,10,11,12].map(g=>'<option '+(g===user.grade?'selected':'')+' value="'+g+'">Grade '+g+'</option>').join('')+'</select></div><div><label>Subject</label><select id="subject"></select></div><div><label>Starting difficulty</label><select id="difficulty"><option>easy</option><option selected>medium</option><option>hard</option></select></div><div><label>Questions</label><select id="count"><option value="5">5 questions</option><option value="10">10 questions</option><option value="15">15 questions</option><option value="20">20 questions</option><option value="25">25 questions</option><option value="30">30 questions</option></select></div><div><label>Timed challenge</label><select id="timerPreset"><option value="">No timer — study at your pace</option><option value="5">5 minutes</option><option value="10">10 minutes</option><option value="20">20 minutes</option><option value="30">30 minutes</option><option value="custom">Set my own timer</option></select></div><div id="customTimerBox" class="hidden"><label>My timer in minutes</label><input id="customTimer" type="number" min="1" max="180" step="1" placeholder="1–180 minutes"><small>Choose any whole number from 1 to 180.</small></div></div><label>Topics</label><div id="topics" class="topics"></div><p><b>Question bank:</b> original curriculum-aligned local questions are used when AI is unavailable.</p><button class="primary">Generate my quiz →</button></form></section>';
  function sync(){let g=$('#grade').value, ss=Object.keys(catalog[g]||{});$('#subject').innerHTML=ss.map(s=>'<option>'+s+'</option>').join('');topicOptions()}
  function topicOptions(){let a=catalog[$('#grade').value][$('#subject').value]||[];$('#topics').innerHTML=a.map((t,i)=>'<button type="button" class="topic '+(i===0?'selected':'')+'" data-topic="'+t+'">'+t+'</button>').join('');$$('.topic').forEach(b=>b.onclick=()=>b.classList.toggle('selected'))}
  $('#grade').onchange=sync;$('#subject').onchange=topicOptions;
  $('#timerPreset').onchange=()=>$('#customTimerBox').classList.toggle('hidden',$('#timerPreset').value!=='custom');
  sync();
  $('#quizForm').onsubmit=async e=>{e.preventDefault();let topics=$$('.topic.selected').map(x=>x.dataset.topic);if(!topics.length)return alert('Choose at least one topic.');let choice=$('#timerPreset').value;let timerMinutes=choice==='custom'?Number($('#customTimer').value):(choice?Number(choice):null);if(choice==='custom'&&(!Number.isInteger(timerMinutes)||timerMinutes<1||timerMinutes>180))return alert('Enter a whole number from 1 to 180 minutes.');try{activeQuiz=await api('/api/quizzes',{method:'POST',body:JSON.stringify({grade:+$('#grade').value,subject:$('#subject').value,topics,question_count:+$('#count').value,starting_difficulty:$('#difficulty').value,timer_minutes:timerMinutes})});renderRunner()}catch(err){alert(err.message)}};
};
// Enriched Overview: provide useful study guidance before the first quiz and after progress exists.
dashboard = async function(){
  let r={completed_quizzes:0,average_score:null,improvement:null,topic_breakdown:[]}, rem={show_today:false}, quizzes=[];
  try { [r,rem,quizzes]=await Promise.all([api('/api/reports/daily'),api('/api/reminders/today'),api('/api/quizzes')]); }
  catch(error){ console.warn('Overview data could not be refreshed:',error); }
  const recommended=(catalog[user.grade]?.Mathematics||Object.values(catalog[user.grade]||{})[0]||[]).slice(0,3);
  const active=quizzes.find(q=>q.status==='active'&&q.current_question);
  const firstUse=r.completed_quizzes===0;
  const weakest=r.topic_breakdown[0];
  const tip=firstUse?'Begin with a 5-question quiz, then use the explanation after each answer to learn from mistakes.':(weakest?'Focus on '+weakest.topic+' next: it is currently marked '+weakest.category+' at '+weakest.accuracy+'%.':'Review one topic, solve a fresh example, and explain your method aloud.');
  const completed=quizzes.filter(q=>q.status==='completed').slice(0,3);
  const recent=completed.length?'<div>'+completed.map(q=>'<div class="insight"><span>'+q.subject+' · Grade '+q.grade+'</span><span>'+q.score_percent+'%</span></div>').join('')+'</div>':'<p>Your recent completed quizzes will appear here after your first session.</p>';
  const weekly=Math.min(r.completed_quizzes,5);
  $('#overview').innerHTML=(rem.show_today?'<div class="reminder"><b>DAILY NUDGE</b><br>'+rem.message+' <button id="dismiss" class="small">Not now</button></div>':'')+
    '<h1>Good morning, <em>'+user.name.split(' ')[0]+'</em>.</h1><p>'+ (firstUse?'Start with a short personalised quiz. Your scores, improvement, and topic strengths will appear here after your first session.':'Here is your learning progress so far. Keep practising to strengthen your topic pulse.') +'</p>'+
    '<div class="grid"><div class="metric"><span>Completed quizzes</span><strong>'+r.completed_quizzes+'</strong></div><div class="metric"><span>Average score</span><strong>'+(r.average_score??'—')+(r.average_score!==null?'%':'')+'</strong></div><div class="metric"><span>Improvement</span><strong>'+(r.improvement===null?'—':(r.improvement>0?'+':'')+r.improvement+'%')+'</strong></div></div>'+
    '<section class="section"><div class="row"><h2>Today’s study guide</h2><button id="reminderSettings" class="logout">Daily reminder</button></div><p>'+tip+'</p><p><b>Weekly rhythm:</b> '+weekly+' of 5 short quizzes completed. Small, regular sessions build confidence.</p></section>'+
    '<section class="section"><div class="row"><h2>'+ (active?'Continue your quiz':'Your next study step') +'</h2></div>'+(active?'<p>Your '+active.subject+' quiz is waiting at '+active.current_question.topic+'.</p><button id="resumeQuiz" class="primary">Resume quiz →</button>':'<p>Suggested starting topics for Grade '+user.grade+': <b>'+recommended.join(' · ')+'</b></p><button id="quickStart" class="primary">Create my first quiz →</button>')+'</section>'+
    '<section class="section"><h2>Recent quiz activity</h2>'+recent+'</section><section class="section"><h2>Topic pulse</h2>'+insights(r.topic_breakdown)+'</section>';
  if($('#dismiss'))$('#dismiss').onclick=async()=>{await api('/api/reminders/dismiss',{method:'POST'});dashboard()};
  if($('#reminderSettings'))$('#reminderSettings').onclick=reminderSettings;
  if($('#quickStart'))$('#quickStart').onclick=()=>{renderBuilder();show('builder')};
  if($('#resumeQuiz'))$('#resumeQuiz').onclick=()=>{activeQuiz=active;renderRunner()};
};
</script></body></html>'''


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return PAGE


if __name__ == "__main__":
    import uvicorn

    # Keep direct execution simple and reliable on Windows and VS Code. Use the
    # documented ``uvicorn main:app --reload`` command when hot reloading is needed.
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)
    
