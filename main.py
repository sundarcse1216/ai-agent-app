from core.controller import Controller
from database.setup_company_database import setup_company_database
from database.setup_events_database import setup_events_database


def banner():
    print("=" * 60)
    print("        🤖 Smart AI Multi-Agent Assistant")
    print("=" * 60)
    print()
    print("Available Features")
    print("-----------------------------")
    print("🌤 Weather Assistant")
    print("🗄 Database Assistant")
    print("🎯 Smart Recommendations")
    print("📚 Knowledge Search")
    print("🎨 AI Image Studio")
    print("💬 Conversational Assistant")
    print()
    print("Type 'exit' to quit.")
    print()


def main():
    setup_company_database()
    setup_events_database()
    controller = Controller()

    banner()

    while True:

        query = input("👤 User : ")

        if query.lower() == "exit":
            print()
            print("=" * 60)
            print("👋 Thank you for using the Smart AI Multi-Agent Assistant!")
            print("🙏 Have a great day. Goodbye!")
            print("=" * 60)
            break

        response = controller.process(query)

        print("🤖 Assistant:", response)


if __name__ == "__main__":
    main()
