from crewai import Task


def create_tasks(agents, user_request, products):
    requirement_task = Task(
        description=f"""
Analyze this customer shopping request:

{user_request}

Extract and organize:
- category
- product
- maximum budget
- condition
- delivery city
- payment method
- delivery deadline
- ingredients
- specifications

Return a clean structured interpretation.
""",
        expected_output="Structured shopping requirements.",
        agent=agents["requirement"],
    )

    search_task = Task(
        description=f"""
Based on the shopping request below, analyze the supplied product data.

USER REQUEST:
{user_request}

PRODUCT DATA:
{products}

Identify products that appear relevant to the request.

Do not invent products, prices, delivery times or payment methods.
""",
        expected_output="List of potentially relevant products from supplied data.",
        agent=agents["search"],
    )

    filter_task = Task(
        description=f"""
User request:
{user_request}

Candidate products:
{products}

Apply these filters:
1. Maximum budget
2. Category
3. Product condition
4. Delivery city
5. Payment method
6. Delivery deadline
7. Ingredients
8. Specifications

Return only products that satisfy the available evidence.
Clearly indicate unavailable information.
""",
        expected_output="Filtered matching products.",
        agent=agents["filter"],
    )

    comparison_task = Task(
        description="""
Compare the filtered products.

Compare:
- Product name
- Source
- Price
- Discount
- Delivery
- Payment method
- Condition
- Ingredients
- Specifications
- Product URL

Create a clear comparison.
""",
        expected_output="Structured product comparison.",
        agent=agents["comparison"],
    )

    recommendation_task = Task(
        description="""
Create the final shopping response based only on the comparison.

Include:
- Matching products
- Price comparison
- Delivery information
- Payment options
- Important specifications
- Ingredients where available
- Product links
- Missing/unverified information

Do not invent facts.
""",
        expected_output="Final shopping comparison and recommendation response.",
        agent=agents["recommendation"],
    )

    return [
        requirement_task,
        search_task,
        filter_task,
        comparison_task,
        recommendation_task,
    ]
