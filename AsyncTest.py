import asyncio
from py_compile import main

class Greeter:
    def __init__(self, name):
        self.name = name

    async def greet(self):
        await asyncio.sleep(1)  # Simulate an asynchronous operation
        return f"Hello, {self.name}!"
    
    
    async def main():
        greeter = Greeter("World")
        greeting = await greeter.greet()
        print(greeting)
        
        
asyncio.run(main())