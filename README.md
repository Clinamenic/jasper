# Jasper

Jasper is a Terminal User Interface (TUI) application for managing and uploading markdown files to Arweave. It provides a clean, intuitive interface for selecting, previewing, and managing your markdown content files.

## Features

- **File Browser**: Navigate and select markdown files from your directory structure
- **File Preview**: View markdown file contents and metadata in real-time
- **Arweave Integration**: Upload selected files to the Arweave network
- **Batch Operations**: Select and upload multiple files at once
- **Test Mode**: Simulate uploads without actually sending files to Arweave
- **Status Tracking**: Monitor upload status and transaction history

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Clinamenic/jasper.git
   cd jasper
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:

   ```bash
   python jasper.py
   ```

2. Navigation:

   - Use arrow keys to navigate the file tree
   - Press `Enter` or `Space` to select/deselect files
   - Press `q` to quit the application
   - Press `r` to refresh the file tree
   - Press `Ctrl+t` to toggle test mode

3. File Selection:

   - Selected files are marked with a filled circle (●)
   - Unselected files are marked with an empty circle (○)
   - Directories containing selected files show selection status

4. Arweave Upload:
   - Select files for upload
   - Review the estimated cost and wallet balance
   - Click "Confirm Upload" or press `c` to start the upload process
   - Monitor progress in the status area

## Directory Structure

The application will look for markdown files (`.md`, `.markdown`) in the parent directory of where it's installed. For example:

```
your_content_root/
  ├── some_markdown.md
  ├── another_markdown.md
  └── jasper/
      └── jasper.py
```

## Configuration

- Arweave wallet configuration:
  - Place your Arweave wallet JSON file as `.wallet.json` in the jasper directory
  - The wallet file should contain your Arweave JWK (JSON Web Key)
  - This file is gitignored by default for security
- Log files are stored in `jasper/jasper.log`
- Test mode can be toggled to simulate uploads without actual blockchain transactions

Example directory structure with wallet:

```
your_content_root/
  ├── some_markdown.md
  ├── another_markdown.md
  └── jasper/
      ├── jasper.py
      └── .wallet.json  # Your Arweave wallet file
```

## Development

To contribute to Jasper:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Terminal with Unicode support for proper display of UI elements

## License

[Add your license information here]
