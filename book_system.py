class BookManager:
    def __init__(self):
        self.books = {}  # 字典存储，key为书名，value为书籍信息字典

    def add_book(self):
        """添加书籍"""
        title = input("请输入书名: ").strip()
        if title in self.books:
            print(f"错误: 书籍《{title}》已存在！")
            return
        author = input("请输入作者: ").strip()
        year = input("请输入出版年份: ").strip()
        isbn = input("请输入ISBN: ").strip()
        self.books[title] = {
            "author": author,
            "year": year,
            "isbn": isbn
        }
        print(f"成功添加书籍《{title}》")

    def delete_book(self):
        """删除书籍"""
        title = input("请输入要删除的书名: ").strip()
        if title in self.books:
            del self.books[title]
            print(f"成功删除书籍《{title}》")
        else:
            print(f"错误: 未找到书籍《{title}》")

    def search_book(self):
        """查找书籍"""
        title = input("请输入要查找的书名: ").strip()
        if title in self.books:
            info = self.books[title]
            print(f"书名: {title}")
            print(f"作者: {info['author']}")
            print(f"出版年份: {info['year']}")
            print(f"ISBN: {info['isbn']}")
        else:
            print(f"未找到书籍《{title}》")

    def update_book(self):
        """修改书籍信息"""
        title = input("请输入要修改的书名: ").strip()
        if title not in self.books:
            print(f"错误: 未找到书籍《{title}》")
            return
        print("请输入新的信息（直接回车表示不修改）:")
        author = input(f"作者 ({self.books[title]['author']}): ").strip()
        year = input(f"出版年份 ({self.books[title]['year']}): ").strip()
        isbn = input(f"ISBN ({self.books[title]['isbn']}): ").strip()
        if author:
            self.books[title]["author"] = author
        if year:
            self.books[title]["year"] = year
        if isbn:
            self.books[title]["isbn"] = isbn
        print(f"成功修改书籍《{title}》的信息")

    def list_books(self):
        """列出所有书籍"""
        if not self.books:
            print("当前没有书籍")
            return
        print("\n当前所有书籍:")
        print("-" * 40)
        for title, info in self.books.items():
            print(f"《{title}》 - 作者: {info['author']}, 年份: {info['year']}, ISBN: {info['isbn']}")
        print("-" * 40)

    def show_menu(self):
        """显示菜单"""
        print("\n" + "=" * 30)
        print("      图书管理系统")
        print("=" * 30)
        print("1. 添加书籍")
        print("2. 删除书籍")
        print("3. 查找书籍")
        print("4. 修改书籍")
        print("5. 列出所有书籍")
        print("6. 退出系统")
        print("=" * 30)

    def run(self):
        """运行系统"""
        while True:
            self.show_menu()
            choice = input("请选择操作 (1-6): ").strip()
            if choice == "1":
                self.add_book()
            elif choice == "2":
                self.delete_book()
            elif choice == "3":
                self.search_book()
            elif choice == "4":
                self.update_book()
            elif choice == "5":
                self.list_books()
            elif choice == "6":
                print("感谢使用图书管理系统，再见！")
                break
            else:
                print("无效选择，请输入1-6之间的数字")

if __name__ == "__main__":
    manager = BookManager()
    manager.run()