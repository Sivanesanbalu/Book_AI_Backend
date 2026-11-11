import json
import sqlite3
import random
import os

# Categories list
CATEGORIES = [
    "Comics 💥",
    "Manga 🎌",
    "Fiction 📚",
    "Sci-Fi & Fantasy 🚀",
    "Self-Help 💡",
    "Technology & Code 💻",
    "Mystery & Thriller 🕵️",
    "History & Bio 📜",
    "Business & Finance 💼"
]

STATUSES = ["Completed", "Reading", "Wishlist"]
STATUS_WEIGHTS = [0.45, 0.35, 0.20] # 45% Completed, 35% Reading, 20% Wishlist

COVERS = {
    "Comics 💥": [
        "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1612036782180-6f0b6cd846fe?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1569003339405-ea396a5a8a90?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=300&q=80"
    ],
    "Manga 🎌": [
        "https://images.unsplash.com/photo-1578632767115-351597cf2477?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1563089145-599997674d42?auto=format&fit=crop&w=300&q=80"
    ],
    "Fiction 📚": [
        "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1512820790803-83ca734da794?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1497633762265-9d179a990aa6?auto=format&fit=crop&w=300&q=80"
    ],
    "Sci-Fi & Fantasy 🚀": [
        "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1506466010722-395aa2bef877?auto=format&fit=crop&w=300&q=80"
    ],
    "Self-Help 💡": [
        "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1506880018603-83d5b814b5a6?auto=format&fit=crop&w=300&q=80"
    ],
    "Technology & Code 💻": [
        "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=300&q=80"
    ],
    "Mystery & Thriller 🕵️": [
        "https://images.unsplash.com/photo-1476275466078-4007374efbbe?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?auto=format&fit=crop&w=300&q=80"
    ],
    "History & Bio 📜": [
        "https://images.unsplash.com/photo-1461360370896-922624d12aa1?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1457369804613-52c61a468e7d?auto=format&fit=crop&w=300&q=80"
    ],
    "Business & Finance 💼": [
        "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&w=300&q=80",
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=300&q=80"
    ]
}

# Raw base catalogs
CATALOG_DATA = {
    "Comics 💥": [
        ("The Amazing Spider-Man: Master Planner", "Stan Lee & Steve Ditko"),
        ("Spider-Man: Blue", "Jeph Loeb & Tim Sale"),
        ("Spider-Man: Kraven's Last Hunt", "J.M. DeMatteis"),
        ("Ultimate Spider-Man: Power and Responsibility", "Brian Michael Bendis"),
        ("Spider-Verse", "Dan Slott & Olivier Coipel"),
        ("Batman: Year One", "Frank Miller & David Mazzucchelli"),
        ("Batman: The Dark Knight Returns", "Frank Miller"),
        ("Batman: The Long Halloween", "Jeph Loeb & Tim Sale"),
        ("Batman: The Killing Joke", "Alan Moore & Brian Bolland"),
        ("Batman: Hush", "Jeph Loeb & Jim Lee"),
        ("Batman: Court of Owls", "Scott Snyder & Greg Capullo"),
        ("Watchmen", "Alan Moore & Dave Gibbons"),
        ("V for Vendetta", "Alan Moore & David Lloyd"),
        ("The Sandman: Preludes & Nocturnes", "Neil Gaiman"),
        ("The Sandman: The Doll's House", "Neil Gaiman"),
        ("The Sandman: Season of Mists", "Neil Gaiman"),
        ("Superman: Red Son", "Mark Millar"),
        ("All-Star Superman", "Grant Morrison & Frank Quitely"),
        ("Superman: For All Seasons", "Jeph Loeb & Tim Sale"),
        ("Kingdom Come", "Mark Waid & Alex Ross"),
        ("Marvels", "Kurt Busiek & Alex Ross"),
        ("Infinity Gauntlet", "Jim Starlin & George Perez"),
        ("Secret Wars", "Jonathan Hickman & Esad Ribic"),
        ("Civil War", "Mark Millar & Steve McNiven"),
        ("X-Men: Days of Future Past", "Chris Claremont & John Byrne"),
        ("X-Men: Dark Phoenix Saga", "Chris Claremont & John Byrne"),
        ("Astonishing X-Men: Gifted", "Joss Whedon & John Cassaday"),
        ("Daredevil: Born Again", "Frank Miller & David Mazzucchelli"),
        ("Daredevil: The Man Without Fear", "Frank Miller & John Romita Jr."),
        ("Saga: Volume 1", "Brian K. Vaughan & Fiona Staples"),
        ("The Walking Dead: Days Gone Bye", "Robert Kirkman"),
        ("Invincible: Family Matters", "Robert Kirkman"),
        ("Hellboy: Seed of Destruction", "Mike Mignola"),
        ("Sin City: The Hard Goodbye", "Frank Miller"),
        ("Scott Pilgrim's Precious Little Life", "Bryan Lee O'Malley"),
        ("Locke & Key: Welcome to Lovecraft", "Joe Hill & Gabriel Rodriguez"),
        ("The Adventures of Tintin: The Secret of the Unicorn", "Herge"),
        ("The Adventures of Tintin: Red Rackham's Treasure", "Herge"),
        ("The Adventures of Tintin: Destination Moon", "Herge"),
        ("The Adventures of Tintin: Explorers on the Moon", "Herge"),
        ("Asterix the Gaul", "Rene Goscinny & Albert Uderzo"),
        ("Asterix and Cleopatra", "Rene Goscinny & Albert Uderzo"),
        ("Asterix in Britain", "Rene Goscinny & Albert Uderzo"),
        ("Maus: A Survivor's Tale", "Art Spiegelman"),
        ("Persepolis", "Marjane Satrapi"),
        ("Y: The Last Man - Unmanned", "Brian K. Vaughan"),
        ("Preacher: Gone to Texas", "Garth Ennis & Steve Dillon"),
        ("The Boys: The Name of the Game", "Garth Ennis"),
        ("Spawn: Origins", "Todd McFarlane"),
        ("Iron Man: Extremis", "Warren Ellis & Adi Granov")
    ],
    "Manga 🎌": [
        ("One Piece: Romance Dawn", "Eiichiro Oda"),
        ("One Piece: Buggy the Clown", "Eiichiro Oda"),
        ("One Piece: The Legend Begins", "Eiichiro Oda"),
        ("One Piece: Alabasta", "Eiichiro Oda"),
        ("One Piece: Skypiea", "Eiichiro Oda"),
        ("One Piece: Water Seven", "Eiichiro Oda"),
        ("One Piece: Enies Lobby", "Eiichiro Oda"),
        ("One Piece: Marineford War", "Eiichiro Oda"),
        ("One Piece: Wano Country", "Eiichiro Oda"),
        ("One Piece: Egghead Arc", "Eiichiro Oda"),
        ("Naruto: The Uzumaki Barrage", "Masashi Kishimoto"),
        ("Naruto: Chunin Exam Finals", "Masashi Kishimoto"),
        ("Naruto: Valley of the End", "Masashi Kishimoto"),
        ("Naruto Shippuden: Tale of Jiraiya the Gallant", "Masashi Kishimoto"),
        ("Naruto Shippuden: Pain's Assault", "Masashi Kishimoto"),
        ("Naruto Shippuden: The Fourth Shinobi World War", "Masashi Kishimoto"),
        ("Bleach: The Substitute", "Tite Kubo"),
        ("Bleach: Soul Society Sneak Entry", "Tite Kubo"),
        ("Bleach: The Rescue", "Tite Kubo"),
        ("Bleach: Thousand-Year Blood War", "Tite Kubo"),
        ("Death Note: Boredom", "Tsugumi Ohba & Takeshi Obata"),
        ("Death Note: Confluence", "Tsugumi Ohba & Takeshi Obata"),
        ("Death Note: Hard Run", "Tsugumi Ohba & Takeshi Obata"),
        ("Attack on Titan: To You, in 2000 Years", "Hajime Isayama"),
        ("Attack on Titan: The Fall of Shiganshina", "Hajime Isayama"),
        ("Attack on Titan: The Uprising", "Hajime Isayama"),
        ("Attack on Titan: Return to Shiganshina", "Hajime Isayama"),
        ("Attack on Titan: The Rumbling", "Hajime Isayama"),
        ("Dragon Ball: The Secret of the Dragon Balls", "Akira Toriyama"),
        ("Dragon Ball Z: The Saiyan Saga", "Akira Toriyama"),
        ("Dragon Ball Z: Freeza Saga", "Akira Toriyama"),
        ("Dragon Ball Z: Cell Games", "Akira Toriyama"),
        ("Dragon Ball Z: Majin Buu Saga", "Akira Toriyama"),
        ("Jujutsu Kaisen: Ryomen Sukuna", "Gege Akutami"),
        ("Jujutsu Kaisen: Shibuya Incident", "Gege Akutami"),
        ("Jujutsu Kaisen: Culling Game", "Gege Akutami"),
        ("Demon Slayer: Cruelty", "Koyoharu Gotouge"),
        ("Demon Slayer: Mugen Train", "Koyoharu Gotouge"),
        ("Demon Slayer: Entertainment District", "Koyoharu Gotouge"),
        ("Chainsaw Man: Dog & Chainsaw", "Tatsuki Fujimoto"),
        ("Chainsaw Man: Gun Devil Arc", "Tatsuki Fujimoto"),
        ("Berserk: The Black Swordsman", "Kentaro Miura"),
        ("Berserk: Golden Age Arc", "Kentaro Miura"),
        ("Tokyo Ghoul: Tragedy", "Sui Ishida"),
        ("Fullmetal Alchemist: The Two Alchemists", "Hiromu Arakawa"),
        ("Hunter x Hunter: The Departure", "Yoshihiro Togashi"),
        ("Hunter x Hunter: Chimera Ant Arc", "Yoshihiro Togashi"),
        ("My Hero Academia: Origin", "Kohei Horikoshi"),
        ("Spy x Family: Mission 1", "Tatsuya Endo"),
        ("JoJo's Bizarre Adventure: Phantom Blood", "Hirohiko Araki"),
        ("JoJo's Bizarre Adventure: Stardust Crusaders", "Hirohiko Araki")
    ],
    "Fiction 📚": [
        ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling"),
        ("Harry Potter and the Chamber of Secrets", "J.K. Rowling"),
        ("Harry Potter and the Prisoner of Azkaban", "J.K. Rowling"),
        ("Harry Potter and the Goblet of Fire", "J.K. Rowling"),
        ("Harry Potter and the Order of the Phoenix", "J.K. Rowling"),
        ("Harry Potter and the Half-Blood Prince", "J.K. Rowling"),
        ("Harry Potter and the Deathly Hallows", "J.K. Rowling"),
        ("The Fellowship of the Ring", "J.R.R. Tolkien"),
        ("The Two Towers", "J.R.R. Tolkien"),
        ("The Return of the King", "J.R.R. Tolkien"),
        ("1984", "George Orwell"),
        ("Animal Farm", "George Orwell"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("Pride and Prejudice", "Jane Austen"),
        ("The Catcher in the Rye", "J.D. Salinger"),
        ("The Alchemist", "Paulo Coelho"),
        ("Life of Pi", "Yann Martel"),
        ("The Kite Runner", "Khaled Hosseini"),
        ("A Thousand Splendid Suns", "Khaled Hosseini"),
        ("The Book Thief", "Markus Zusak"),
        ("Jane Eyre", "Charlotte Bronte"),
        ("Wuthering Heights", "Emily Bronte"),
        ("Little Women", "Louisa May Alcott"),
        ("Brave New World", "Aldous Huxley"),
        ("Lord of the Flies", "William Golding"),
        ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
        ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),
        ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("The Brothers Karamazov", "Fyodor Dostoevsky"),
        ("War and Peace", "Leo Tolstoy"),
        ("Anna Karenina", "Leo Tolstoy"),
        ("The Picture of Dorian Gray", "Oscar Wilde"),
        ("The Metamorphosis", "Franz Kafka"),
        ("The Trial", "Franz Kafka"),
        ("The Stranger", "Albert Camus"),
        ("The Plague", "Albert Camus"),
        ("Frankenstein", "Mary Shelley"),
        ("Dracula", "Bram Stoker"),
        ("The Old Man and the Sea", "Ernest Hemingway"),
        ("For Whom the Bell Tolls", "Ernest Hemingway"),
        ("Of Mice and Men", "John Steinbeck"),
        ("The Grapes of Wrath", "John Steinbeck"),
        ("Slaughterhouse-Five", "Kurt Vonnegut"),
        ("Catch-22", "Joseph Heller"),
        ("Fahrenheit 451", "Ray Bradbury"),
        ("Things Fall Apart", "Chinua Achebe"),
        ("Beloved", "Toni Morrison"),
        ("The Road", "Cormac McCarthy"),
        ("No Country for Old Men", "Cormac McCarthy")
    ],
    "Sci-Fi & Fantasy 🚀": [
        ("Dune", "Frank Herbert"),
        ("Dune Messiah", "Frank Herbert"),
        ("Children of Dune", "Frank Herbert"),
        ("God Emperor of Dune", "Frank Herbert"),
        ("Foundation", "Isaac Asimov"),
        ("Foundation and Empire", "Isaac Asimov"),
        ("Second Foundation", "Isaac Asimov"),
        ("Neuromancer", "William Gibson"),
        ("Count Zero", "William Gibson"),
        ("Mona Lisa Overdrive", "William Gibson"),
        ("The Hobbit", "J.R.R. Tolkien"),
        ("The Silmarillion", "J.R.R. Tolkien"),
        ("Ender's Game", "Orson Scott Card"),
        ("Speaker for the Dead", "Orson Scott Card"),
        ("The Three-Body Problem", "Cixin Liu"),
        ("The Dark Forest", "Cixin Liu"),
        ("Death's End", "Cixin Liu"),
        ("Snow Crash", "Neal Stephenson"),
        ("The Diamond Age", "Neal Stephenson"),
        ("Hyperion", "Dan Simmons"),
        ("The Fall of Hyperion", "Dan Simmons"),
        ("Do Androids Dream of Electric Sheep?", "Philip K. Dick"),
        ("Ubik", "Philip K. Dick"),
        ("The Man in the High Castle", "Philip K. Dick"),
        ("2001: A Space Odyssey", "Arthur C. Clarke"),
        ("Rendezvous with Rama", "Arthur C. Clarke"),
        ("The Left Hand of Darkness", "Ursula K. Le Guin"),
        ("A Wizard of Earthsea", "Ursula K. Le Guin"),
        ("The Dispossessed", "Ursula K. Le Guin"),
        ("The Name of the Wind", "Patrick Rothfuss"),
        ("The Wise Man's Fear", "Patrick Rothfuss"),
        ("The Way of Kings", "Brandon Sanderson"),
        ("Words of Radiance", "Brandon Sanderson"),
        ("Oathbringer", "Brandon Sanderson"),
        ("Mistborn: The Final Empire", "Brandon Sanderson"),
        ("The Well of Ascension", "Brandon Sanderson"),
        ("The Hero of Ages", "Brandon Sanderson"),
        ("A Game of Thrones", "George R.R. Martin"),
        ("A Clash of Kings", "George R.R. Martin"),
        ("A Storm of Swords", "George R.R. Martin"),
        ("The Martian", "Andy Weir"),
        ("Project Hail Mary", "Andy Weir"),
        ("Red Rising", "Pierce Brown"),
        ("Golden Son", "Pierce Brown"),
        ("Morning Star", "Pierce Brown"),
        ("Leviathan Wakes", "James S.A. Corey"),
        ("Caliban's War", "James S.A. Corey"),
        ("Abaddon's Gate", "James S.A. Corey"),
        ("Solaris", "Stanislaw Lem"),
        ("Starship Troopers", "Robert A. Heinlein")
    ],
    "Self-Help 💡": [
        ("Atomic Habits", "James Clear"),
        ("Deep Work", "Cal Newport"),
        ("Digital Minimalism", "Cal Newport"),
        ("So Good They Can't Ignore You", "Cal Newport"),
        ("Think and Grow Rich", "Napoleon Hill"),
        ("Rich Dad Poor Dad", "Robert T. Kiyosaki"),
        ("The 7 Habits of Highly Effective People", "Stephen R. Covey"),
        ("The Power of Now", "Eckhart Tolle"),
        ("A New Earth", "Eckhart Tolle"),
        ("Ikigai: The Japanese Secret to a Long and Happy Life", "Hector Garcia & Francesc Miralles"),
        ("Can't Hurt Me", "David Goggins"),
        ("Never Finished", "David Goggins"),
        ("The 4-Hour Workweek", "Timothy Ferriss"),
        ("The Subtle Art of Not Giving a F*ck", "Mark Manson"),
        ("Everything Is F*cked: A Book About Hope", "Mark Manson"),
        ("Mindset: The New Psychology of Success", "Carol S. Dweck"),
        ("Grit: The Power of Passion and Perseverance", "Angela Duckworth"),
        ("Make Your Bed", "William H. McRaven"),
        ("Man's Search for Meaning", "Viktor E. Frankl"),
        ("Daring Greatly", "Brene Brown"),
        ("The Gifts of Imperfection", "Brene Brown"),
        ("Thinking, Fast and Slow", "Daniel Kahneman"),
        ("Quiet: The Power of Introverts", "Susan Cain"),
        ("Essentialism: The Disciplined Pursuit of Less", "Greg McKeown"),
        ("Effortless", "Greg McKeown"),
        ("The Miracle Morning", "Hal Elrod"),
        ("Eat That Frog!", "Brian Tracy"),
        ("Awaken the Giant Within", "Tony Robbins"),
        ("Unlimited Power", "Tony Robbins"),
        ("The 5 AM Club", "Robin Sharma"),
        ("The Monk Who Sold His Ferrari", "Robin Sharma"),
        ("Meditations", "Marcus Aurelius"),
        ("Letters from a Stoic", "Seneca"),
        ("Discourses and Selected Writings", "Epictetus"),
        ("The Daily Stoic", "Ryan Holiday"),
        ("Ego Is the Enemy", "Ryan Holiday"),
        ("The Obstacle Is the Way", "Ryan Holiday"),
        ("Courage Is Calling", "Ryan Holiday"),
        ("Discipline Is Destiny", "Ryan Holiday"),
        ("Show Your Work!", "Austin Kleon"),
        ("Steal Like an Artist", "Austin Kleon"),
        ("Keep Going", "Austin Kleon"),
        ("You Are a Badass", "Jen Sincero"),
        ("The Mountain Is You", "Brianna Wiest"),
        ("101 Essays That Will Change The Way You Think", "Brianna Wiest"),
        ("Start With Why", "Simon Sinek"),
        ("Find Your Why", "Simon Sinek"),
        ("The Infinite Game", "Simon Sinek"),
        ("Leaders Eat Last", "Simon Sinek"),
        ("How to Win Friends and Influence People", "Dale Carnegie")
    ],
    "Technology & Code 💻": [
        ("Clean Code: A Handbook of Agile Software Craftsmanship", "Robert C. Martin"),
        ("The Clean Coder", "Robert C. Martin"),
        ("Clean Architecture", "Robert C. Martin"),
        ("The Pragmatic Programmer", "David Thomas & Andrew Hunt"),
        ("Designing Data-Intensive Applications", "Martin Kleppmann"),
        ("Python Crash Course", "Eric Matthes"),
        ("Automate the Boring Stuff with Python", "Al Sweigart"),
        ("Fluent Python", "Luciano Ramalho"),
        ("Effective Python", "Brett Slatkin"),
        ("Introduction to Algorithms (CLRS)", "Thomas H. Cormen et al."),
        ("Algorithms", "Robert Sedgewick & Kevin Wayne"),
        ("Grokking Algorithms", "Aditya Bhargava"),
        ("Refactoring: Improving the Design of Existing Code", "Martin Fowler"),
        ("Patterns of Enterprise Application Architecture", "Martin Fowler"),
        ("Domain-Driven Design", "Eric Evans"),
        ("Structure and Interpretation of Computer Programs", "Harold Abelson & Gerald Jay Sussman"),
        ("Head First Design Patterns", "Eric Freeman & Elisabeth Robson"),
        ("Design Patterns: Elements of Reusable Object-Oriented Software", "Gang of Four"),
        ("You Don't Know JS Yet: Get Started", "Kyle Simpson"),
        ("Eloquent JavaScript", "Marijn Haverbeke"),
        ("JavaScript: The Good Parts", "Douglas Crockford"),
        ("Cracking the Coding Interview", "Gayle Laakmann McDowell"),
        ("System Design Interview", "Alex Xu"),
        ("System Design Interview – An Insider's Guide: Volume 2", "Alex Xu & Sahn Lam"),
        ("Database Internals", "Alex Petrov"),
        ("Understanding Distributed Systems", "Roberto Vitillo"),
        ("Site Reliability Engineering", "Betsy Beyer et al."),
        ("Building Microservices", "Sam Newman"),
        ("Monolith to Microservices", "Sam Newman"),
        ("Accelerate: Building and Scaling High Performing Tech", "Nicole Forsgren et al."),
        ("The Phoenix Project", "Gene Kim et al."),
        ("The Unicorn Project", "Gene Kim"),
        ("The DevOps Handbook", "Gene Kim et al."),
        ("Deep Learning", "Ian Goodfellow, Yoshua Bengio & Aaron Courville"),
        ("Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", "Aurelien Geron"),
        ("Pattern Recognition and Machine Learning", "Christopher M. Bishop"),
        ("Speech and Language Processing", "Dan Jurafsky & James H. Martin"),
        ("Natural Language Processing with Transformers", "Lewis Tunstall et al."),
        ("Programming Rust", "Jim Blandy & Jason Orendorff"),
        ("The Rust Programming Language", "Steve Klabnik & Carol Nichols"),
        ("The Go Programming Language", "Alan A. A. Donovan & Brian W. Kernighan"),
        ("Learning Go", "Jon Bodner"),
        ("C Programming Language", "Brian W. Kernighan & Dennis M. Ritchie"),
        ("Modern Operating Systems", "Andrew S. Tanenbaum"),
        ("Computer Networks", "Andrew S. Tanenbaum"),
        ("Operating Systems: Three Easy Pieces", "Remzi H. Arpaci-Dusseau"),
        ("Linux Kernel Development", "Robert Love"),
        ("TCP/IP Illustrated, Volume 1", "W. Richard Stevens"),
        ("Kubernetes in Action", "Marko Luksa"),
        ("Docker Deep Dive", "Nigel Poulton")
    ],
    "Mystery & Thriller 🕵️": [
        ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        ("The Hound of the Baskervilles", "Arthur Conan Doyle"),
        ("A Study in Scarlet", "Arthur Conan Doyle"),
        ("The Sign of the Four", "Arthur Conan Doyle"),
        ("The Valley of Fear", "Arthur Conan Doyle"),
        ("The Da Vinci Code", "Dan Brown"),
        ("Angels & Demons", "Dan Brown"),
        ("Inferno", "Dan Brown"),
        ("The Lost Symbol", "Dan Brown"),
        ("Gone Girl", "Gillian Flynn"),
        ("Sharp Objects", "Gillian Flynn"),
        ("Dark Places", "Gillian Flynn"),
        ("And Then There Were None", "Agatha Christie"),
        ("Murder on the Orient Express", "Agatha Christie"),
        ("The Murder of Roger Ackroyd", "Agatha Christie"),
        ("Death on the Nile", "Agatha Christie"),
        ("The A.B.C. Murders", "Agatha Christie"),
        ("The Girl with the Dragon Tattoo", "Stieg Larsson"),
        ("The Girl Who Played with Fire", "Stieg Larsson"),
        ("The Girl Who Kicked the Hornet's Nest", "Stieg Larsson"),
        ("The Silence of the Lambs", "Thomas Harris"),
        ("Red Dragon", "Thomas Harris"),
        ("Hannibal", "Thomas Harris"),
        ("Shutter Island", "Dennis Lehane"),
        ("Mystic River", "Dennis Lehane"),
        ("The Silent Patient", "Alex Michaelides"),
        ("The Maidens", "Alex Michaelides"),
        ("The Fury", "Alex Michaelides"),
        ("Big Little Lies", "Liane Moriarty"),
        ("The Husband's Secret", "Liane Moriarty"),
        ("Nine Perfect Strangers", "Liane Moriarty"),
        ("The Girl on the Train", "Paula Hawkins"),
        ("Into the Water", "Paula Hawkins"),
        ("A Slow Fire Burning", "Paula Hawkins"),
        ("In the Woods", "Tana French"),
        ("The Likeness", "Tana French"),
        ("Faithful Place", "Tana French"),
        ("The Guest List", "Lucy Foley"),
        ("The Paris Apartment", "Lucy Foley"),
        ("The Hunting Party", "Lucy Foley"),
        ("One of Us Is Lying", "Karen M. McManus"),
        ("Two Can Keep a Secret", "Karen M. McManus"),
        ("Verity", "Colleen Hoover"),
        ("Before I Go to Sleep", "S.J. Watson"),
        ("The Woman in the Window", "A.J. Finn"),
        ("The Snowman", "Jo Nesbo"),
        ("The Bat", "Jo Nesbo"),
        ("Defending Jacob", "William Landay"),
        ("Behind Closed Doors", "B.A. Paris"),
        ("The Couple Next Door", "Shari Lapena")
    ],
    "History & Bio 📜": [
        ("Sapiens: A Brief History of Humankind", "Yuval Noah Harari"),
        ("Homo Deus: A Brief History of Tomorrow", "Yuval Noah Harari"),
        ("21 Lessons for the 21st Century", "Yuval Noah Harari"),
        ("Steve Jobs", "Walter Isaacson"),
        ("Elon Musk", "Walter Isaacson"),
        ("Leonardo da Vinci", "Walter Isaacson"),
        ("Einstein: His Life and Universe", "Walter Isaacson"),
        ("Benjamin Franklin: An American Life", "Walter Isaacson"),
        ("Wings of Fire: An Autobiography", "A.P.J. Abdul Kalam"),
        ("Ignited Minds", "A.P.J. Abdul Kalam"),
        ("India 2020", "A.P.J. Abdul Kalam"),
        ("Ponniyin Selvan: First Floods", "Kalki Krishnamurthy"),
        ("Ponniyin Selvan: Whirlwind", "Kalki Krishnamurthy"),
        ("Ponniyin Selvan: Killer Sword", "Kalki Krishnamurthy"),
        ("Ponniyin Selvan: The Crown", "Kalki Krishnamurthy"),
        ("Ponniyin Selvan: Pinnacle of Sacrifice", "Kalki Krishnamurthy"),
        ("Sivagamiyin Sabatham", "Kalki Krishnamurthy"),
        ("Parthiban Kanavu", "Kalki Krishnamurthy"),
        ("Alexander Hamilton", "Ron Chernow"),
        ("Titan: The Life of John D. Rockefeller", "Ron Chernow"),
        ("Grant", "Ron Chernow"),
        ("Washington: A Life", "Ron Chernow"),
        ("The Diary of a Young Girl", "Anne Frank"),
        ("A Promised Land", "Barack Obama"),
        ("Dreams from My Father", "Barack Obama"),
        ("Becoming", "Michelle Obama"),
        ("Long Walk to Freedom", "Nelson Mandela"),
        ("Shoe Dog: A Memoir by the Creator of Nike", "Phil Knight"),
        ("Educated", "Tara Westover"),
        ("Born a Crime: Stories from a South African Childhood", "Trevor Noah"),
        ("Guns, Germs, and Steel", "Jared Diamond"),
        ("Collapse: How Societies Choose to Fail or Succeed", "Jared Diamond"),
        ("The Silk Roads: A New History of the World", "Peter Frankopan"),
        ("SPQR: A History of Ancient Rome", "Mary Beard"),
        ("Rubicon: The Last Years of the Roman Republic", "Tom Holland"),
        ("Persian Fire", "Tom Holland"),
        ("Dynasty: The Rise and Fall of the House of Caesar", "Tom Holland"),
        ("The Wright Brothers", "David McCullough"),
        ("1776", "David McCullough"),
        ("John Adams", "David McCullough"),
        ("Churchill: A Life", "Martin Gilbert"),
        ("The Splendid and the Vile", "Erik Larson"),
        ("The Devil in the White City", "Erik Larson"),
        ("Dead Wake: The Last Crossing of the Lusitania", "Erik Larson"),
        ("Team of Rivals: The Political Genius of Abraham Lincoln", "Doris Kearns Goodwin"),
        ("Leadership: In Turbulent Times", "Doris Kearns Goodwin"),
        ("Midnight in Chernobyl", "Adam Higginbotham"),
        ("The Anarchy: The Relentless Rise of the East India Company", "William Dalrymple"),
        ("White Mughals", "William Dalrymple"),
        ("The Last Mughal", "William Dalrymple")
    ],
    "Business & Finance 💼": [
        ("Zero to One: Notes on Startups", "Peter Thiel & Blake Masters"),
        ("The Psychology of Money", "Morgan Housel"),
        ("Same as Ever: A Guide to What Never Changes", "Morgan Housel"),
        ("Good to Great", "Jim Collins"),
        ("Built to Last", "Jim Collins & Jerry I. Porras"),
        ("How the Mighty Fall", "Jim Collins"),
        ("The Intelligent Investor", "Benjamin Graham"),
        ("Security Analysis", "Benjamin Graham & David Dodd"),
        ("The Lean Startup", "Eric Ries"),
        ("The Startup Way", "Eric Ries"),
        ("Principles: Life and Work", "Ray Dalio"),
        ("Principles for Dealing with the Changing World Order", "Ray Dalio"),
        ("Hooked: How to Build Habit-Forming Products", "Nir Eyal"),
        ("Blue Ocean Strategy", "W. Chan Kim & Renee Mauborgne"),
        ("Rework", "Jason Fried & David Heinemeier Hansson"),
        ("Remote: Office Not Required", "Jason Fried & David Heinemeier Hansson"),
        ("It Doesn't Have to Be Crazy at Work", "Jason Fried & David Heinemeier Hansson"),
        ("The Hard Thing About Hard Things", "Ben Horowitz"),
        ("What You Do Is Who You Are", "Ben Horowitz"),
        ("High Output Management", "Andrew S. Grove"),
        ("Only the Paranoid Survive", "Andrew S. Grove"),
        ("Measure What Matters", "John Doerr"),
        ("The Innovator's Dilemma", "Clayton M. Christensen"),
        ("How Will You Measure Your Life?", "Clayton M. Christensen"),
        ("Crossing the Chasm", "Geoffrey A. Moore"),
        ("Thinking in Bets", "Annie Duke"),
        ("Fooled by Randomness", "Nassim Nicholas Taleb"),
        ("The Black Swan", "Nassim Nicholas Taleb"),
        ("Antifragile: Things That Gain from Disorder", "Nassim Nicholas Taleb"),
        ("Skin in the Game", "Nassim Nicholas Taleb"),
        ("Poor Charlie's Almanack", "Charles T. Munger"),
        ("The Warren Buffett Way", "Robert G. Hagstrom"),
        ("The Snowball: Warren Buffett and the Business of Life", "Alice Schroeder"),
        ("Common Stocks and Uncommon Profits", "Philip A. Fisher"),
        ("One Up On Wall Street", "Peter Lynch"),
        ("Beating the Street", "Peter Lynch"),
        ("Liar's Poker", "Michael Lewis"),
        ("The Big Short", "Michael Lewis"),
        ("Flash Boys", "Michael Lewis"),
        ("Moneyball", "Michael Lewis"),
        ("Bad Blood: Secrets and Lies in a Silicon Valley Startup", "John Carreyrou"),
        ("Shoe Dog", "Phil Knight"),
        ("Creativity, Inc.", "Ed Catmull & Amy Wallace"),
        ("Ride of a Lifetime", "Robert Iger"),
        ("Trillion Dollar Coach", "Eric Schmidt et al."),
        ("No Rules Rules: Netflix and the Culture of Reinvention", "Reed Hastings & Erin Meyer"),
        ("Working Backwards: Insights, Stories, and Secrets from Inside Amazon", "Colin Bryar & Bill Carr"),
        ("The Everything Store: Jeff Bezos and the Age of Amazon", "Brad Stone"),
        ("Amazon Unbound", "Brad Stone"),
        ("Superintelligence: Paths, Dangers, Strategies", "Nick Bostrom")
    ]
}

def generate_1000_books():
    books_1000 = []
    
    # Target counts per category: ~111 each to hit 1000
    target_per_cat = {
        "Comics 💥": 115,
        "Manga 🎌": 115,
        "Fiction 📚": 115,
        "Sci-Fi & Fantasy 🚀": 115,
        "Self-Help 💡": 110,
        "Technology & Code 💻": 110,
        "Mystery & Thriller 🕵️": 110,
        "History & Bio 📜": 110,
        "Business & Finance 💼": 105
    }
    
    global_id = 1
    
    for cat, target in target_per_cat.items():
        base_list = CATALOG_DATA.get(cat, [])
        cat_covers = COVERS.get(cat, COVERS["Fiction 📚"])
        
        # Add primary base books
        for idx, (title, author) in enumerate(base_list):
            status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
            rating = random.choice([4, 5, 5, 4, 3])
            cover = cat_covers[idx % len(cat_covers)]
            overview = f"A globally acclaimed work in {cat} by {author}, exploring iconic themes, character mastery, and unforgettable narratives."
            
            books_1000.append({
                "book_id": f"bk_{global_id:04d}",
                "title": title,
                "author": author,
                "genre": cat,
                "cover_url": cover,
                "overview": overview,
                "confidence": round(random.uniform(0.92, 0.99), 3),
                "status": status,
                "rating": rating,
                "notes": f"Catalog entry #{global_id} in {cat}"
            })
            global_id += 1
            if len([b for b in books_1000 if b["genre"] == cat]) >= target:
                break
                
        # Fill remaining with numbered editions, chronicles, and series extensions
        current_cat_count = len([b for b in books_1000 if b["genre"] == cat])
        counter = 1
        while current_cat_count < target:
            base_title, base_author = base_list[counter % len(base_list)]
            suffix = f"Vol. {counter + 1}" if "💥" in cat or "🎌" in cat else f"Part {counter + 1}: The Chronicles" if "🚀" in cat or "📚" in cat else f"Edition {counter + 1}: Advanced Insights"
            title = f"{base_title} ({suffix})"
            status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
            rating = random.choice([4, 5, 5, 4, 3])
            cover = cat_covers[(counter + 2) % len(cat_covers)]
            overview = f"Continuing the legendary {cat} journey by {base_author}. A masterclass narrative exploring deep dilemmas, insights, and perspectives."
            
            books_1000.append({
                "book_id": f"bk_{global_id:04d}",
                "title": title,
                "author": base_author,
                "genre": cat,
                "cover_url": cover,
                "overview": overview,
                "confidence": round(random.uniform(0.91, 0.98), 3),
                "status": status,
                "rating": rating,
                "notes": f"Edition #{counter + 1}"
            })
            global_id += 1
            current_cat_count += 1
            counter += 1

    print(f"Generated {len(books_1000)} books across {len(target_per_cat)} categories!")
    return books_1000

def seed_database(books):
    db_path = "/Users/sivanesanbalu/.gemini/antigravity/scratch/Book_AI_Backend/data/library.db"
    json_path = "/Users/sivanesanbalu/.gemini/antigravity/scratch/Book_AI_Backend/data/books.json"
    
    # 1. Save to JSON master file
    with open(json_path, "w") as f:
        json.dump(books, f, indent=2)
    print(f"✅ Saved {len(books)} books to {json_path}")
    
    # 2. Insert into SQLite library.db for 'default_user' and demo
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Clear old user_books for default_user to replace with full 1,000 catalog
    cur.execute("DELETE FROM user_books WHERE uid = 'default_user'")
    
    inserted = 0
    for b in books:
        cur.execute("""
        INSERT INTO user_books (uid, book_id, title, author, genre, cover_url, overview, confidence, status, rating, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "default_user",
            b["book_id"],
            b["title"],
            b["author"],
            b["genre"],
            b["cover_url"],
            b["overview"],
            b["confidence"],
            b["status"],
            b["rating"],
            b["notes"]
        ))
        inserted += 1
        
    conn.commit()
    
    # Verify count
    cur.execute("SELECT count(*) FROM user_books WHERE uid = 'default_user'")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT status, count(*) FROM user_books WHERE uid = 'default_user' GROUP BY status")
    stats = cur.fetchall()
    
    conn.close()
    print(f"✅ Successfully seeded SQLite with {total} books!")
    print(f"📊 Status Breakdown: {stats}")

if __name__ == "__main__":
    books = generate_1000_books()
    seed_database(books)
