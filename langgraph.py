from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ActionGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json
import requests
from PIL import Image
from io import BytesIO

# Initialize LLMs
#llm = ChatOpenAI(model="gpt-4-turbo-preview")
llm = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key="ur Open API Key")


# Example Database (replace with your actual database or API)
freelancer_db = [
    {"name": "Alice", "skills": ["web design", "UI/UX"], "portfolio": ["image1.jpg", "image2.jpg"]},
    {"name": "Bob", "skills": ["graphic design", "logo design"], "portfolio": ["logo1.jpg", "logo2.jpg"]},
    {"name": "Charlie", "skills": ["video editing", "animation"], "portfolio": ["video1.mp4", "video2.mp4"]},
]

# Tools
@tool
def search_freelancers(query: str) -> str:
    """Searches for freelancers based on skills and portfolio."""
    results = []
    for freelancer in freelancer_db:
        if any(skill.lower() in query.lower() for skill in freelancer["skills"]):
            results.append(freelancer["name"])
    return json.dumps(results)

@tool
def extract_requirements(text: str) -> str:
    """Extracts project requirements from text."""
    messages = [
        SystemMessage(content="Extract key project requirements."),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages).content
    return response

@tool
def analyze_image(image_url: str) -> str:
    """Analyzes an image and extracts relevant information."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Placeholder: Replace with actual image analysis logic
        return f"Image analysis: {image.format}, {image.size}"
    except Exception as e:
        return f"Error analyzing image: {e}"

@tool
def communicate(message: str) -> str:
    """Simulates communication between client and freelancer."""
    return f"Communication: {message}"

@tool
def manage_project(details: str) -> str:
    """Manages project details and timelines."""
    return f"Project details: {details}"

tools = [search_freelancers, extract_requirements, analyze_image, communicate, manage_project]
tool_executor = ToolExecutor(tools)

# Nodes
def extract_requirements_node(state):
    messages = state["messages"]
    tool_input = {"text": messages[-1].content}
    response = tool_executor.invoke({"tool_name": "extract_requirements", "tool_input": tool_input})
    return {"messages": messages + [AIMessage(content=response["output"])]}

def analyze_image_node(state):
    messages = state["messages"]
    image_url = messages[-1].content #assuming the last message is image url.
    tool_input = {"image_url": image_url}
    response = tool_executor.invoke({"tool_name": "analyze_image", "tool_input": tool_input})
    return {"messages": messages + [AIMessage(content=response["output"])]}

def search_freelancers_node(state):
    messages = state["messages"]
    tool_input = {"query": messages[-1].content}
    response = tool_executor.invoke({"tool_name": "search_freelancers", "tool_input": tool_input})
    return {"messages": messages + [AIMessage(content=response["output"])]}

def communicate_node(state):
    messages = state["messages"]
    tool_input = {"message": messages[-1].content}
    response = tool_executor.invoke({"tool_name": "communicate", "tool_input": tool_input})
    return {"messages": messages + [AIMessage(content=response["output"])]}

def manage_project_node(state):
    messages = state["messages"]
    tool_input = {"details": messages[-1].content}
    response = tool_executor.invoke({"tool_name": "manage_project", "tool_input": tool_input})
    return {"messages": messages + [AIMessage(content=response["output"])]}

def router_node(state):
    messages = state["messages"]
    last_message = messages[-1].content

    # Logic to determine the next node based on the last message
    if "image" in last_message.lower() and "http" in last_message.lower():
        return {"next_node": "analyze_image", "messages": messages}  # Route to image analysis
    elif "freelancer" in last_message.lower() or "skill" in last_message.lower():
        return {"next_node": "search_freelancers", "messages": messages} # Route to freelancer search
    elif "project details" in last_message.lower():
        return {"next_node": "manage_project", "messages": messages} # Route to project management
    else:
        return {"next_node": "extract_requirements", "messages": messages} # Default to requirement extraction.

# Graph
workflow = StateGraph(lambda state: state["next_node"] if "next_node" in state else "extract_requirements")
workflow.add_node("extract_requirements", extract_requirements_node)
workflow.add_node("analyze_image", analyze_image_node)
workflow.add_node("search_freelancers", search_freelancers_node)
workflow.add_node("communicate", communicate_node)
workflow.add_node("manage_project", manage_project_node)
workflow.add_node("router", router_node)

workflow.add_edge("extract_requirements", "router")
workflow.add_edge("analyze_image", "router")
workflow.add_edge("search_freelancers", "router")
workflow.add_edge("communicate", "router")
workflow.add_edge("manage_project", "router")

workflow.add_edge("router", "analyze_image", condition = lambda state: state.get("next_node") == "analyze_image")
workflow.add_edge("router", "search_freelancers", condition = lambda state: state.get("next_node") == "search_freelancers")
workflow.add_edge("router", "manage_project", condition = lambda state: state.get("next_node") == "manage_project")
workflow.add_edge("router", "extract_requirements", condition = lambda state: state.get("next_node") == "extract_requirements")

workflow.set_entry_point("extract_requirements")

graph = workflow.compile()

# Example Usage
inputs = {"messages": [HumanMessage(content="I need a website designed. Here is an example: https://www.example.com/image.jpg")]}
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}':")
        print(value.get("messages", value))