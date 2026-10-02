from crewai import Crew, Process

from agents.shopping_agents import create_agents
from agents.shopping_tasks import create_tasks


def run_shopping_crew(user_request, products):
    agents = create_agents()

    tasks = create_tasks(
        agents=agents,
        user_request=user_request,
        products=products,
    )

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    return result
