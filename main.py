import chainlit as cl
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel ,function_tool
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent



# @cl.on_message
# async def main(message: cl.Message):
#     # Your custom logic goes here...

#     # Send a response back to the user
#     await cl.Message(
#         content=f"Received: {message.content}",
#     ).send()

CSV_PATH = "availablity.csv"

def load_data():
    return pd.read_csv(CSV_PATH)

def save_data(df):
    df.to_csv(CSV_PATH, index=False)

# Get availability for a given date
@function_tool
def get_availability(date: str) -> str:
    """
            Returns a list of available room types for a given date.

        Args:
            date (str): The date to check availability for (format: YYYY-MM-DD).

        Returns:
            str: A formatted string listing available room types and counts,
                or a message indicating no availability or invalid date.

    """
    df = load_data()
    day_rooms = df[df['date'] == date]

    if day_rooms.empty:
        return f"No availability data for {date}."

    available_rooms = day_rooms[day_rooms['available'] > 0]
    if available_rooms.empty:
        return f"No rooms available on {date}."

    result = [f"{row.room_type.title()} ({row.available} available)" for _, row in available_rooms.iterrows()]
    return f"Available rooms on {date}:\n" + "\n".join(result)

# Book a room if available
@function_tool
def book_room(date: str, room_type: str, guests: int) -> str:
    """
    Attempts to book a room of the specified type on a given date.

Args:
    date (str): The booking date (format: YYYY-MM-DD).
    room_type (str): The type of room to book (e.g., 'single', 'double', 'suite').
    guests (int): Number of guests for the booking.

Returns:
    str: A confirmation message if booking is successful,
            or an error message if the room is unavailable or does not exist.
    """
    df = load_data()
    match = (df['date'] == date) & (df['room_type'].str.lower() == room_type.lower())
    room_row = df[match]

    if room_row.empty:
        return f"No {room_type} room available on {date}."

    idx = room_row.index[0]
    if df.at[idx, 'available'] > 0:
        df.at[idx, 'available'] -= 1
        save_data(df)
        return f"✅ Booked a {room_type.title()} room for {guests} guest(s) on {date}."
    else:
        return f"❌ No {room_type.title()} rooms left on {date}."



load_dotenv(find_dotenv())
api_key = os.getenv('OpenAI_api_key')

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.aimlapi.com/v1",
)

model = OpenAIChatCompletionsModel(
    model="gpt-4",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent: Agent = Agent(
    name="Receptionist", 
    instructions="You are a helpful hotel receptionist. You can help with booking rooms, checking availability regarding specific date user provides, and answering questions about the hotel.",
    tools=[get_availability, book_room], 
    model=model
    )


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message("Hi! This ABC Hotels Receptionist which can book hotel room, answer common queries and check hotel rooms availiblity from hotel's database for you!!").send()

@cl.on_message
async def handle(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({
        "role": "user",
        "content": message.content
    })

    msg = cl.Message(content="")

    # Format history as a string to avoid passing raw dictionaries
    def format_history(history):
        formatted = []
        for entry in history:
            role = entry["role"].title()
            content = entry["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    result = Runner.run_streamed(agent, format_history(history), run_config=config)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            response = event.data.delta
            if isinstance(response, str):
                await msg.stream_token(response)
            elif isinstance(response, dict) and "content" in response:
                await msg.stream_token(response["content"])
            else:
                print(f"Unexpected response format: {response}")
        
        elif event.type == "final_response":
            if hasattr(event.data, "content") and isinstance(event.data.content, str):
                await msg.stream_token(event.data.content)
            elif isinstance(event.data, dict) and "content" in event.data:
                await msg.stream_token(event.data["content"])
            else:
                print(f"Unexpected final response format: {event.data}")

    # Store only the content in history
    history.append({
        "role": "receptionist",
        "content": msg.content
    })
    cl.user_session.set("history", history)

    await msg.update()




