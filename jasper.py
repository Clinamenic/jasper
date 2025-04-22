#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypedDict, cast
import time
import select
import re
import shutil

import frontmatter
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Markdown,
    Static,
    Tree,
    TabbedContent,
    TabPane,
)
from textual.widgets.tree import TreeNode
import rich
from rich.text import Text

# Import the Arweave functions
import modules.jasper_arweave as jasper_arweave


# Types - Keep core types needed by the app itself
# Arweave-specific types are now in jasper_arweave
class DirectoryState(TypedDict):
    expanded: bool
    selected_files: List[str]


# Constants - Keep app-level constants
# Arweave-specific constants (DEFAULT_TAGS, TEST_MODE) moved to jasper_arweave

# Setup logger - Keep main app logger setup
LOG_PATH = Path("logs/jasper.log")  # Updated log path to logs directory
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(LOG_PATH, mode="a"),  # Append mode
    ],
)
# Configure root logger and add handler for jasper_arweave module
logger = logging.getLogger("jasper")  # Main app logger name
# Ensure logs from jasper_arweave go to the same file
arweave_logger = logging.getLogger("jasper.arweave")
arweave_logger.addHandler(logging.FileHandler(LOG_PATH, mode="a"))
arweave_logger.setLevel(logging.DEBUG)  # Match main logger level
arweave_logger.propagate = False  # Prevent duplicate logs in root logger


# --- Core TUI Widgets (Largely Unchanged) ---


class FileNode(Static):
    """A widget to display a file in the tree with checkbox."""

    def __init__(self, file_path: Path, is_selected: bool = False):
        super().__init__()
        self.file_path = file_path
        self.is_selected = is_selected
        self.update_display()

    def update_display(self) -> None:
        """Update the display of the file node."""
        checkbox = "[x]" if self.is_selected else "[ ]"
        self.update(f"{checkbox} {self.file_path.name}")

    def toggle_selection(self) -> None:
        """Toggle the selection state of the file."""
        self.is_selected = not self.is_selected
        self.update_display()


class ContentTree(Tree):
    """Custom tree widget for content directory."""

    def __init__(self, root_path: Path):
        super().__init__(
            "Files", data=root_path
        )  # Changed from "Content Directory" to "Files"
        self.root_path = root_path
        self.queued_files: Set[Path] = set()
        # Get the jasper directory path to exclude it
        self.jasper_dir = Path(__file__).resolve().parent

    def _get_files_recursive(self, directory: Path) -> List[Path]:
        """Recursively find all .md/.markdown files in a directory."""
        file_paths = []
        try:
            for path in directory.iterdir():
                # Skip the jasper directory and hidden directories
                if path.is_dir():
                    if path == self.jasper_dir or path.name.startswith("."):
                        continue
                    file_paths.extend(self._get_files_recursive(path))
                elif path.is_file() and path.suffix.lower() in (".md", ".markdown"):
                    file_paths.append(path)
        except OSError as e:
            logger.error(f"Error accessing directory {directory}: {e}")
        return sorted(file_paths, key=lambda p: p.name.lower())

    def _find_child_file_nodes(self, parent_node: TreeNode) -> List[TreeNode]:
        """Recursively find all file TreeNodes under a given directory node."""
        file_nodes = []
        for child in parent_node.children:
            # Ensure child.data exists before checking is_dir()
            if child.data:
                if not child.data.is_dir():  # It's a file node
                    file_nodes.append(child)
                else:  # It's a directory node
                    file_nodes.extend(self._find_child_file_nodes(child))
        return file_nodes

    def toggle_directory_selection(self, node: TreeNode) -> None:
        """Toggle selection for all files within a directory node."""
        path = cast(Path, node.data)
        child_file_nodes = self._find_child_file_nodes(node)

        if not child_file_nodes:
            return

        child_file_paths = [
            cast(Path, file_node.data) for file_node in child_file_nodes
        ]

        all_selected = all(p in self.queued_files for p in child_file_paths)

        if all_selected:
            self.queued_files.difference_update(child_file_paths)
            node.label = f"○ {path.name}"  # Unchecked (hollow white)
            for file_node in child_file_nodes:
                file_node.label = (
                    f"○ {cast(Path, file_node.data).name}"  # Unchecked (hollow white)
                )
        else:
            self.queued_files.update(child_file_paths)
            node.label = f"● {path.name}"  # Checked (Filled Circle)
            for file_node in child_file_nodes:
                file_node.label = (
                    f"● {cast(Path, file_node.data).name}"  # Checked (Filled Circle)
                )

        self.refresh()

    def load_directory(
        self, directory: Path, parent: Optional[TreeNode] = None
    ) -> None:
        """Load directory contents into the tree."""
        if parent is None:
            parent = self.root
            parent.data = directory

        try:
            # Get all paths in the directory, excluding jasper dir and hidden files
            paths = sorted(
                [
                    p
                    for p in directory.iterdir()
                    if not p.name.startswith(".") and p != self.jasper_dir
                ],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )

            for path in paths:
                if path.is_dir():
                    # Check if directory contains any markdown files (recursively)
                    contained_files = self._get_files_recursive(path)
                    if contained_files:
                        initial_label = f"○ {path.name}"
                        node = parent.add(initial_label, data=path, expand=False)
                        self.load_directory(path, node)
                elif path.suffix.lower() in (".md", ".markdown"):
                    initial_label = f"○ {path.name}"  # Unchecked by default
                    if path in self.queued_files:
                        initial_label = f"● {path.name}"  # Show as checked if in queue
                    parent.add_leaf(initial_label, data=path)

        except OSError as e:
            logger.error(f"Error loading directory {directory}: {e}")
            # Add an error node to show the problem in the UI
            parent.add_leaf(f"Error: {str(e)}", data=None)

    def toggle_file_selection(self, node: TreeNode) -> None:
        """Toggle file selection and update the queue."""
        if node.data and not node.data.is_dir():  # Check node.data exists
            path = cast(Path, node.data)

            if path in self.queued_files:
                self.queued_files.remove(path)
                node.label = f"○ {path.name}"  # Unchecked (hollow white)
            else:
                self.queued_files.add(path)
                node.label = f"● {path.name}"  # Checked (Filled Circle)

            self.refresh()

    def get_queued_files(self) -> List[Path]:
        """Get the list of queued files."""
        return list(self.queued_files)


class FilePreview(Vertical):
    """Widget to preview file content and status (potentially across protocols)."""

    def __init__(self):
        super().__init__(id="file-preview-container")
        self.current_file: Optional[Path] = None
        self._arweave_index_data: Optional[List[Dict]] = None  # Cache loaded index

    def compose(self) -> ComposeResult:
        """Compose the file preview widget."""
        yield Static("No file selected", id="file-path")
        yield Static("UUID: N/A", id="file-uuid")
        yield Static("Title: N/A", id="file-title")
        # Generic status, can be updated based on protocol context later
        yield Static("Status: N/A", id="file-status")
        yield Markdown("", id="file-content")

    def update_preview(self, file_path: Optional[Path]) -> None:
        """Update the preview with file content and basic status."""
        self.current_file = file_path

        if file_path is None or not file_path.exists():
            self.query_one("#file-path", Static).update("No file selected")
            self.query_one("#file-uuid", Static).update("UUID: N/A")
            self.query_one("#file-title", Static).update("Title: N/A")
            self.query_one("#file-status", Static).update("Status: N/A")
            self.query_one("#file-content", Markdown).update("")
            return

        self.query_one("#file-path", Static).update(f"Selected: {file_path.name}")

        try:
            # Load raw file content
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Load with frontmatter for metadata extraction
            post = frontmatter.loads(raw_content)
            uuid_value = post.get("uuid", "N/A")
            title = post.get("title", "Untitled")

            self.query_one("#file-uuid", Static).update(f"UUID: {uuid_value}")
            self.query_one("#file-title", Static).update(f"Title: {title}")

            # Check Arweave status (as default/primary status for now)
            # Load index data if not already cached
            if self._arweave_index_data is None:
                self._arweave_index_data = jasper_arweave.load_arweave_index()

            status = jasper_arweave.get_arweave_status(
                uuid_value, index_data=self._arweave_index_data
            )
            # TODO: Update status based on active tab later?
            self.query_one("#file-status", Static).update(f"Arweave Status: {status}")

            # Display raw content
            self.query_one("#file-content", Markdown).update(
                f"```markdown\n{raw_content}\n```"
            )

        except frontmatter.FrontmatterError as e:
            logger.error(f"Error parsing frontmatter for {file_path}: {str(e)}")
            self.query_one("#file-content", Markdown).update(
                f"Error loading frontmatter: {str(e)}"
            )
            self.query_one("#file-status", Static).update(
                "Status: Error parsing frontmatter"
            )
        except Exception as e:
            logger.error(f"Error previewing file {file_path}: {str(e)}")
            self.query_one("#file-content", Markdown).update(
                f"Error loading file preview: {str(e)}"
            )
            self.query_one("#file-status", Static).update(
                "Status: Error loading preview"
            )

    def clear_arweave_index_cache(self) -> None:
        """Clears the cached Arweave index data."""
        self._arweave_index_data = None
        logger.info("Cleared cached Arweave index data in FilePreview.")


# --- Arweave Tab Widgets (Recreated) ---


class ArweaveTagsInfo(Static):
    """Widget to display Arweave default tag information."""

    def __init__(self):
        super().__init__("")
        logger.debug("Initializing ArweaveTagsInfo widget")
        self.update_tags()

    def update_tags(self) -> None:
        """Update the tags display using tags from jasper_arweave."""
        try:
            logger.debug("Updating ArweaveTagsInfo tags display")
            tags_text = "Default Tags: "
            # Use DEFAULT_TAGS from the imported module
            for i, tag in enumerate(jasper_arweave.DEFAULT_TAGS):
                if i > 0:
                    tags_text += ", "
                tags_text += f"{tag['name']}={tag['value']}"
            tags_text += ", UUID=(from frontmatter), Type=(from frontmatter)"
            self.update(tags_text)
            logger.debug(f"Updated ArweaveTagsInfo text: {tags_text}")
        except Exception as e:
            logger.error(f"Error updating ArweaveTagsInfo: {str(e)}")
            # Provide fallback text
            self.update("Default Arweave tags will be applied to uploads.")

    def compose(self) -> ComposeResult:
        """Compose the tags info widget."""
        logger.debug("Composing ArweaveTagsInfo (no children to yield)")
        # Static widgets don't need to yield anything, but the method must be a generator
        if False:  # This makes it a generator that never yields anything
            yield


class ArweaveQueueStats(Static):
    """Widget to display Arweave queue statistics and actions."""

    def __init__(self):
        super().__init__("")
        logger.debug("Initializing ArweaveQueueStats widget")
        self.file_count = 0
        self.total_cost = 0.0
        self.wallet_balance = 0.0

    def compose(self) -> ComposeResult:
        """Compose the queue stats widget with buttons."""
        logger.debug("Composing ArweaveQueueStats widget")
        try:
            with Horizontal(id="arweave-stats-container"):
                yield Static(self._format_stats(), id="arweave-stats-text")
                with Horizontal(id="arweave-stats-buttons"):
                    yield Button(
                        "Confirm Upload", id="arw-confirm-upload", variant="primary"
                    )
                    yield Button(
                        "Refresh Stats", id="arw-refresh-stats"
                    )  # Renamed from back-button
            logger.debug("Successfully composed ArweaveQueueStats widget")
        except Exception as e:
            logger.error(f"Error composing ArweaveQueueStats widget: {str(e)}")

    def _format_stats(self) -> str:
        """Format the statistics text."""
        # Indicate test mode status if active but avoid using rich markup syntax
        test_mode_indicator = " (TEST MODE)" if jasper_arweave.TEST_MODE else ""
        stats_text = (
            f"Queued: {self.file_count} files{test_mode_indicator}\n"
            f"Est. Cost: {self.total_cost:.6f} AR\n"
            f"Balance: {self.wallet_balance:.6f} AR"
        )
        logger.debug(f"ArweaveQueueStats formatted text: {stats_text}")
        return stats_text

    def update_stats(
        self, file_count: int, total_cost: float, wallet_balance: float
    ) -> None:
        """Update queue statistics."""
        logger.debug(
            f"Updating ArweaveQueueStats: files={file_count}, cost={total_cost:.6f}, balance={wallet_balance:.6f}"
        )
        self.file_count = file_count
        self.total_cost = total_cost
        self.wallet_balance = wallet_balance

        if self.is_mounted:
            logger.debug("ArweaveQueueStats is mounted, updating display")
            self.query_one("#arweave-stats-text", Static).update(self._format_stats())
            # Enable/disable confirm button based on balance vs cost (unless in test mode)
            confirm_button = self.query_one("#arw-confirm-upload", Button)
            can_afford = self.wallet_balance >= self.total_cost
            confirm_button.disabled = not (
                can_afford or jasper_arweave.TEST_MODE or file_count == 0
            )
            confirm_button.tooltip = (
                None if confirm_button.disabled else "Upload selected files to Arweave"
            )
            if not can_afford and not jasper_arweave.TEST_MODE and file_count > 0:
                confirm_button.tooltip = f"Insufficient balance ({self.wallet_balance:.6f} AR) for estimated cost ({self.total_cost:.6f} AR)"
        else:
            logger.warning("ArweaveQueueStats is not mounted, cannot update display")


class ArweaveUploadQueue(Container):
    """Widget for Arweave upload queue display."""

    def __init__(self):
        super().__init__()
        logger.debug("Initializing ArweaveUploadQueue widget")
        self._table_display = Static("", id="arweave-queue-table")
        self.queued_files_details: List[Tuple[Path, float, str, str]] = []
        logger.debug("ArweaveUploadQueue widget initialized with empty queue")

    def compose(self) -> ComposeResult:
        """Compose the upload queue widget."""
        logger.debug("Composing ArweaveUploadQueue widget")
        with Container(id="queue-table-container"):
            yield self._table_display
        logger.debug("ArweaveUploadQueue widget composed successfully")

    def update_queue_display(
        self, file_details: List[Tuple[Path, float, str, str]]
    ) -> None:
        """Update the display table with file details and costs."""
        logger.debug(f"Updating ArweaveUploadQueue with {len(file_details)} files")
        self.queued_files_details = file_details

        if not file_details:
            logger.debug("No files in ArweaveUploadQueue, showing empty message")
            self._table_display.update(
                "No files selected for Arweave upload.\nSelect files in the tree to add them to the queue."
            )
            return

        # Create a rich table for display
        table = Table(
            expand=False,
            show_header=True,
            show_lines=True,
            box=rich.box.SIMPLE,
            padding=(0, 1),
        )

        # Add columns with specific widths and styles - match mica.py's widths
        table.add_column("Est. Cost", width=12, style="#ee7c6b")
        table.add_column("File", width=40, style="white")
        table.add_column("Type", width=15, style="#ee7c6b")
        table.add_column("UUID", width=36, style="#ee7c6b")

        # Add rows
        total_cost = 0.0
        for path, cost, file_type, uuid_value in file_details:
            total_cost += cost
            uuid_short = uuid_value[:32] + "..." if len(uuid_value) > 35 else uuid_value
            table.add_row(f"{cost:.6f} AR", path.name, file_type, uuid_short)

        # Add total row with a different style
        table.add_row(f"{total_cost:.6f} AR", "TOTAL COST", "", "", style="#d05340")

        try:
            self._table_display.update(table)
            logger.debug("Successfully updated queue display")
        except Exception as e:
            logger.error(f"Error updating queue display: {str(e)}")
            # Fallback to a simple display if rich formatting fails
            self._table_display.update(
                f"Error displaying table. {len(file_details)} files queued, total cost: {total_cost:.6f} AR"
            )


# --- Main Application Class (Integrating Tabs) ---


class JasperApp(App):
    """The main Jasper TUI application with protocol tabs."""

    # Set app title to display in header
    TITLE = "J A S P E R"

    # Use inline CSS instead of CSS_PATH
    CSS = """
    $accent: #d05340; 
    $text-color: #fcfcd8;
    $background-color: #300801; 
    $accent-dark: #8d1f0e 85%; 
    $accent-light: #ee7c6b;
    $hatch-shade: "░" #d05340 60%;
    $hatch-shade2: "░" cyan 60%;


* {
    scrollbar-color: $accent-dark;
    scrollbar-color-hover: $accent;
    scrollbar-color-active: $accent;
    background: black 0%;
}

Screen {
    color: $text-color;
    background: black;
    hatch: $hatch-shade;
    padding: 0 2;
}

Header {
    dock: top;
    height: 10%;
    min-height: 1;
    max-height: 3;
    background: $accent-dark 70%;
    color: $text-color;
    margin: 1 0;
    text-style: bold;
    content-align: center middle;
}

Footer {
    dock: bottom;
    height: 10%;
    min-height: 1;
    max-height: 3;
    background: $accent-dark;
    color: $text-color;
    align: center middle;
    margin: 1 0;
}

#app-grid {
    layout: grid;
    grid-size: 2;
    grid-columns: 1fr 2fr;
    grid-rows: 1fr;
    grid-gutter: 1 2;
    width: 100%;
    height: 100%;
    hatch: $hatch-shade;
}

#left-panel {
    layout: grid;
    grid-size: 1;
    grid-rows: 1fr 1fr;
    grid-gutter: 1 0;
    height: 100%;
    hatch: $hatch-shade;
}

#right-panel {
    height: 100%;
    width: 100%;
    hatch: $hatch-shade;
}

ContentTree {
    border: thick $accent-dark;
    background: $background-color;
    height: 100%;
    padding: 0 1;
}

Tree:focus .tree--cursor {
    background: $accent; /* Use variable */
    color: black;
}

FilePreview {
    border: thick $accent-dark;
    background: $background-color;
    height: 100%;
    padding: 0 1;
}

#file-preview-container {
    padding: 0 1;
    height: 100%;
    layout: grid;
    grid-size: 1;
    grid-rows: auto auto auto auto 100%;
}

#file-content {
    overflow-y: auto;
    padding: 0;
    background: $background-color;
}

TabbedContent {
    height: 100%;
    width: 100%;
    background: $background-color;
    border: thick $accent-dark;
}

TabPane {
    padding: 0;
    height: 100%;
    width: 100%;
}

/* Arweave Tab specific styles */
#arweave-tab {
    padding: 0;
    height: 100%;
    width: 100%;
}

#arweave-content {
    height: 100%;
    width: 100%;
    padding: 0;
    layout: grid;
    grid-size: 1;
    grid-rows: 1fr auto 5;  /* Queue gets 1fr, tags auto, stats fixed 5 */
    grid-gutter: 1;
    hatch: $hatch-shade;
}

ArweaveUploadQueue {
    height: 100%;
}

#queue-table-container {
    overflow-y: auto;  /* Enable vertical scrolling */
    overflow-x: auto;  /* Enable horizontal scrolling */
    height: 100%;  /* Take full height */
    width: 100%;
    padding: 0 1;
    background: $background-color;
    border: thick $accent-dark;
}

#arweave-queue-table {
    background: $background-color;
    color: $text-color;
    width: auto;  /* Allow table to expand naturally */
    height: auto;
}

/* Style the rich table elements */
#arweave-queue-table .rich-table {
    width: auto;  /* Let table size to content */
    color: $text-color;
}

#arweave-queue-table .rich-table-header {
    background: $background-color;
    color: $text-color;
}

ArweaveTagsInfo {
    height: auto;
    min-height: 3;
    border: thick $accent-dark;
    padding: 1;
    background: $background-color;
    color: $text-color;
}

ArweaveQueueStats {
    height: 5;
    border: thick $accent-dark;
    background: $background-color;
}

#arweave-stats-container {
    padding: 0 1;
    align: center middle;
    width: 100%;
}

#arweave-stats-text {
    width: 1fr;
    content-align-vertical: middle;
}

#arweave-stats-buttons {
    width: auto;
    align-horizontal: right;
}

#arweave-stats-buttons Button {
    margin-left: 1;
    min-width: 15;
}

#arw-confirm-upload {
    background: $accent;
    color: $text-color;
    border: none;
    padding: 1;
}

#arw-confirm-upload:hover {
    background: $accent-dark;
}

#arw-confirm-upload:disabled {
    background: $accent 30%;
    color: $text-color 50%;
}

#arw-refresh-stats {
    background: $accent;
    color: $text-color;
    border: none;
    padding: 1;
}

#arw-refresh-stats:hover {
    background: $accent-dark;
}

/* AT Proto placeholder */
#atproto-placeholder {
    padding: 2;
    content-align: center middle;
}
"""

    # Updated bindings for tab navigation and actions
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_tree", "Refresh Tree"),
        Binding("ctrl+t", "toggle_test_mode", "Toggle Arweave Test Mode"),
        Binding("u", "update_arweave_stats", "Update Arweave Stats"),  # Example action
        Binding(
            "c", "confirm_arweave_upload", "Confirm Arweave Upload"
        ),  # Example action
        Binding("d", "debug_ui", "Debug UI"),  # Added for troubleshooting
        # Add bindings for tab switching if needed (Textual defaults might be ok)
    ]

    def __init__(self):
        super().__init__()
        # Get the directory containing jasper.py
        script_dir = Path(__file__).resolve().parent
        # Use parent directory of the jasper directory
        self.content_path = script_dir.parent
        self._active_arweave_task = None  # Track background Arweave tasks
        self._queued_files_for_arweave: List[Path] = []

    def compose(self) -> ComposeResult:
        """Compose the app layout with tabs."""
        yield Header()

        with Container(id="app-grid"):
            with Container(id="left-panel"):
                yield ContentTree(self.content_path)
                yield FilePreview()

            with Container(id="right-panel"):
                with Vertical(id="arweave-content"):
                    # Order matters: Queue table first, then tags, then stats
                    yield ArweaveUploadQueue()  # Large queue table at the top
                    yield ArweaveTagsInfo()  # Small tags section in the middle
                    yield ArweaveQueueStats()  # Stats section at the bottom

        yield Footer()

    def on_mount(self) -> None:
        """Log session start and load tree."""
        try:
            with open(LOG_PATH, "a") as f:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write(f"# JASPER SESSION STARTED ({datetime.now()}) #\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            logger.error(f"Failed to write session separator to log {LOG_PATH}: {e}")

        logger.info("Jasper started")
        logger.debug("Setting up initial UI state")

        self.query_one(ContentTree).load_directory(self.content_path)
        self.query_one(ContentTree).focus()

        # Log the state of the widgets
        logger.debug("Widgets after mount:")
        logger.debug(f"ContentTree: {self.query_one(ContentTree)}")
        logger.debug(f"FilePreview: {self.query_one(FilePreview)}")
        logger.debug(f"ArweaveUploadQueue: {self.query_one(ArweaveUploadQueue)}")
        logger.debug(f"ArweaveTagsInfo: {self.query_one(ArweaveTagsInfo)}")
        logger.debug(f"ArweaveQueueStats: {self.query_one(ArweaveQueueStats)}")

        # Initialize the Arweave widgets explicitly
        try:
            logger.debug("Initializing Arweave UI components")
            self.query_one(ArweaveTagsInfo).update_tags()
            self.query_one(ArweaveUploadQueue).update_queue_display([])
            # Stats will be updated by update_arweave_stats_async
            logger.debug("Arweave UI components initialized")
        except Exception as e:
            logger.error(f"Error initializing Arweave UI components: {str(e)}")

        # Use event loop to call update_arweave_stats_async
        logger.debug("Scheduling initial Arweave stats update")
        asyncio.create_task(self.update_arweave_stats_async())

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Update preview and trigger context update for active tab."""
        logger.debug("on_tree_node_selected called")
        tree = self.query_one(ContentTree)
        node = event.node
        if not node.data or not isinstance(node.data, Path):
            logger.debug("Invalid node data, skipping selection")
            return
        path = cast(Path, node.data)

        if path.is_dir():
            logger.debug(f"Directory selected: {path}")
            tree.toggle_directory_selection(node)
            self.query_one(FilePreview).update_preview(None)
        else:
            logger.debug(f"File selected: {path}")
            tree.toggle_file_selection(node)
            self.query_one(FilePreview).update_preview(path)

        # Update Arweave context
        self._queued_files_for_arweave = tree.get_queued_files()
        logger.debug(f"Queue updated: {len(self._queued_files_for_arweave)} files")

        # Use create_task to update Arweave stats asynchronously
        logger.debug("Scheduling Arweave stats update after selection change")
        asyncio.create_task(self.update_arweave_stats_async())

        logger.debug(
            f"Queued files changed: {[f.name for f in self._queued_files_for_arweave]}"
        )

    async def update_arweave_stats_async(self) -> None:
        """Simple helper to call update_arweave_context with current queued files"""
        logger.debug(
            f"update_arweave_stats_async called, queued files: {len(self._queued_files_for_arweave)}"
        )
        await self.update_arweave_context(self._queued_files_for_arweave)

    async def update_arweave_context(self, files: List[Path]) -> None:
        """Worker to update Arweave queue display and stats."""
        logger.debug(f"update_arweave_context running with {len(files)} files")
        if not self.is_running:
            logger.debug("App not running, skipping update_arweave_context")
            return  # Prevent updates if app is closing

        try:
            arweave_queue_widget = self.query_one(ArweaveUploadQueue)
            arweave_stats_widget = self.query_one(ArweaveQueueStats)

            logger.debug(f"Found queue widget: {arweave_queue_widget}")
            logger.debug(f"Found stats widget: {arweave_stats_widget}")

            if not files:
                logger.debug("No files selected, clearing Arweave widgets")
                try:
                    arweave_queue_widget.update_queue_display([])
                except Exception as e:
                    logger.error(f"Error clearing queue widget: {str(e)}")

                try:
                    balance = await jasper_arweave.get_wallet_balance()
                    arweave_stats_widget.update_stats(0, 0.0, balance)
                except Exception as e:
                    logger.error(f"Error updating stats widget: {str(e)}")

                return

            # Get balance first
            balance = await jasper_arweave.get_wallet_balance()
            logger.debug(f"Arweave wallet balance: {balance:.6f} AR")

            file_details = []
            total_cost = 0.0
            # Estimate costs concurrently? For now, sequentially
            for file_path in files:
                logger.debug(f"Estimating cost for file: {file_path}")
                cost = await jasper_arweave.estimate_upload_cost(file_path)
                total_cost += cost
                uuid_value, file_type = "N/A", "N/A"
                try:
                    post = frontmatter.load(file_path)
                    uuid_value = post.get("uuid", "N/A")
                    file_type = post.get("type", "N/A")
                    logger.debug(
                        f"Loaded frontmatter for {file_path.name}: uuid={uuid_value}, type={file_type}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not parse frontmatter for {file_path.name} during queue update: {str(e)}"
                    )
                file_details.append((file_path, cost, file_type, uuid_value))

            # Update queue widget separately from stats widget so one failing doesn't affect the other
            try:
                logger.debug(
                    f"Updating Arweave queue with {len(file_details)} files, total cost: {total_cost:.6f}"
                )
                arweave_queue_widget.update_queue_display(file_details)
            except Exception as e:
                logger.error(f"Error updating queue widget: {str(e)}")

            try:
                logger.debug(
                    f"Updating Arweave stats with file count: {len(files)}, total cost: {total_cost:.6f}, balance: {balance:.6f}"
                )
                arweave_stats_widget.update_stats(len(files), total_cost, balance)
            except Exception as e:
                logger.error(f"Error updating stats widget: {str(e)}")

        except Exception as e:
            logger.error(f"Error in update_arweave_context: {str(e)}", exc_info=True)

    # --- Actions ---

    def action_refresh_tree(self) -> None:
        """Refresh tree, clear preview, and update Arweave context."""
        tree = self.query_one(ContentTree)
        preview = self.query_one(FilePreview)
        tree.clear()
        tree.load_directory(self.content_path)
        preview.update_preview(None)
        preview.clear_arweave_index_cache()
        # Also refresh Arweave context after tree refresh
        self._queued_files_for_arweave = tree.get_queued_files()
        self.update_arweave_stats_async()
        self.notify("Content tree refreshed")
        logger.info("Content tree refreshed by user.")
        tree.focus()

    def action_toggle_test_mode(self) -> None:
        """Toggle Arweave TEST_MODE."""
        jasper_arweave.TEST_MODE = not jasper_arweave.TEST_MODE
        mode = "enabled" if jasper_arweave.TEST_MODE else "disabled"
        self.notify(f"Arweave TEST MODE {mode}.", title="Test Mode Toggle")
        logger.info(f"Arweave TEST MODE toggled: {mode}")
        # Refresh stats to show indicator/update button state
        self.update_arweave_stats_async()

    def action_update_arweave_stats(self) -> None:
        """Manually trigger update of Arweave stats/queue view."""
        self.notify("Refreshing Arweave stats...")
        asyncio.create_task(self.update_arweave_stats_async())

    def action_confirm_arweave_upload(self) -> None:
        """Confirm and start Arweave upload process."""
        if not self._queued_files_for_arweave:
            self.notify("No files selected for Arweave upload.", severity="warning")
            return

        if self._active_arweave_task and not self._active_arweave_task.done():
            self.notify("An Arweave upload is already in progress.", severity="warning")
            return

        # Perform checks (e.g., balance) - already done in update_arweave_context for button state
        # Could re-check here for safety
        stats_widget = self.query_one(ArweaveQueueStats)
        if (
            not jasper_arweave.TEST_MODE
            and stats_widget.wallet_balance < stats_widget.total_cost
        ):
            self.notify(
                "Insufficient wallet balance for Arweave upload.", severity="error"
            )
            return

        confirm_msg = "Starting Arweave upload..."
        if jasper_arweave.TEST_MODE:
            confirm_msg = "Starting Arweave upload simulation (TEST MODE)..."
        self.notify(confirm_msg)
        logger.info(
            f"User confirmed Arweave upload for {len(self._queued_files_for_arweave)} files."
        )

        # Run upload directly as an async task
        asyncio.create_task(
            self.run_arweave_upload(self._queued_files_for_arweave.copy())
        )

    async def run_arweave_upload(self, files_to_upload: List[Path]) -> None:
        """Worker function to execute the Arweave upload."""
        results = await jasper_arweave.upload_files_to_arweave(
            files_to_upload=files_to_upload,
            app_notify_callback=self.notify,  # Pass app's notify method
        )
        # Upload finished, maybe refresh Arweave status and clear queue visually?
        logger.info(f"Arweave upload worker finished. Results: {results}")
        # Refresh file preview status and Arweave tab context
        self.query_one(FilePreview).clear_arweave_index_cache()
        current_preview_file = self.query_one(FilePreview).current_file
        if current_preview_file:
            self.query_one(FilePreview).update_preview(current_preview_file)
        # Refresh queue/stats display
        await self.update_arweave_context(self._queued_files_for_arweave)

    def action_debug_ui(self) -> None:
        """Debug helper to log the UI widget tree."""
        self.notify("Logging UI widget tree to console")
        logger.info("--- DEBUG UI WIDGET TREE ---")

        # Log the right panel and arweave content
        right_panel = self.query_one("#right-panel")
        logger.info(f"Right panel: {right_panel}")
        logger.info(f"Right panel children: {list(right_panel.children)}")

        # Log Arweave content
        try:
            arweave_content = self.query_one("#arweave-content")
            logger.info(f"Arweave content: {arweave_content}")
            logger.info(f"Arweave content children: {list(arweave_content.children)}")

            # Detailed look at ArweaveUploadQueue
            queue_widget = self.query_one(ArweaveUploadQueue)
            logger.info(f"ArweaveUploadQueue: {queue_widget}")
            logger.info(f"  ID: {queue_widget.id}")
            logger.info(f"  Visible: {queue_widget.visible}")
            logger.info(f"  Display: {queue_widget.display}")
            logger.info(f"  Size: {queue_widget.size}")
            logger.info(f"  Children count: {len(list(queue_widget.children))}")
            for i, child in enumerate(queue_widget.children):
                logger.info(f"  Child {i}: {child}")

            # Check the table display widget
            try:
                table_display = self.query_one("#arweave-queue-table")
                logger.info(f"Queue table display: {table_display}")
                logger.info(f"  Visible: {table_display.visible}")
                logger.info(f"  Text: {repr(table_display.renderable)[:100]}...")
            except Exception as e:
                logger.error(f"Error finding queue table display: {str(e)}")

            # Check other widgets too
            for widget_class in [ArweaveTagsInfo, ArweaveQueueStats]:
                try:
                    widget = self.query_one(widget_class)
                    logger.info(f"Found {widget_class.__name__}: {widget}")
                    logger.info(f"  Visible: {widget.visible}")
                    logger.info(f"  Size: {widget.size}")
                except Exception as e:
                    logger.error(f"Error finding {widget_class.__name__}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in debug_ui: {str(e)}")

    # --- Button Handling ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the Arweave tab."""
        button_id = event.button.id

        if button_id == "arw-confirm-upload":
            self.action_confirm_arweave_upload()
        elif button_id == "arw-refresh-stats":
            self.action_update_arweave_stats()

    # --- Unmount ---

    def on_unmount(self) -> None:
        """Log session stop."""
        logger.info("Jasper stopped")
        # Consider cancelling active workers? Textual might handle this.


if __name__ == "__main__":
    try:
        app = JasperApp()
        app.run()
    except Exception as e:
        # Log fatal exceptions before exiting
        logger.exception("Jasper encountered a fatal error and had to exit.")
